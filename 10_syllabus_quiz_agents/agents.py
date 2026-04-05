# agents.py
# Three-agent workflow: course Q&A (RAG + tools), quiz authoring, progress logging
# Tim Fraser

# Agent 1: syllabus outline + retrieved excerpts as hints; Ollama teaches (general + course-scoped).
# Agent 3 calls update_progress when the learner earns 80% or higher.

import json
import os
import re
from collections.abc import Callable
from datetime import datetime, timezone

import requests

from functions import (
    CHAT_URL,
    DEFAULT_MODEL,
    REQUEST_TIMEOUT,
    agent_run,
    ensure_ollama_available,
)
from syllabus_rag import (
    format_course_outline,
    parse_markdown_sections,
    search_sections,
    sections_context_for_llm,
)
from workspace import load_progress, load_quizzes, read_syllabus_text, save_progress, save_quizzes

# Configuration: override with environment variable if needed
MODEL = os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)

# 1. Multi-step chat with tools (retrieve context, then answer) ##################


def chat_with_tool_loop(
    messages: list[dict],
    tools: list[dict],
    tool_dispatch: dict[str, Callable[..., object]],
    model: str = MODEL,
    max_rounds: int = 6,
) -> str:
    """
    Repeatedly call Ollama until the model returns plain text (no tool_calls).

    tool_dispatch maps function name -> Python callable(**args).
    """
    ensure_ollama_available()
    msgs: list[dict] = [dict(m) for m in messages]

    for _ in range(max_rounds):
        body = {
            "model": model,
            "messages": msgs,
            "tools": tools,
            "stream": False,
            "options": {"num_predict": 1024},
        }
        response = requests.post(CHAT_URL, json=body, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        result = response.json()
        msg = result["message"]
        msgs.append(msg)

        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            return (msg.get("content") or "").strip()

        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            raw_args = tc["function"].get("arguments", {})
            fn_args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            if fn_name not in tool_dispatch:
                tool_out = {"error": f"unknown tool {fn_name}"}
            else:
                tool_out = tool_dispatch[fn_name](**fn_args)
            payload = json.dumps(tool_out) if isinstance(tool_out, (dict, list)) else str(tool_out)
            msgs.append({"role": "tool", "content": payload})

    return (msgs[-1].get("content") if isinstance(msgs[-1], dict) else "") or ""


# 2. Agent 1 — course Q&A (syllabus as hints; model supplies teaching content) ###

ROLE_CONTENT_TUTOR = """You are a tutor for this course. Each user message includes:

1) Course map — the full list of syllabus section titles (how the course is organized and directed).
2) Related excerpts — text pulled from sections that best match the student's question. Use these
   for vocabulary, emphasis, and what this offering cares about — not as the only information you
   may use.

Explain clearly using your general knowledge as needed. Match the level and themes suggested by the
course map and excerpts. If helpful, you may briefly separate "In this course / per your materials"
from a broader explanation.

For binding admin (grades, policies, due dates, what is required reading): only state what appears
explicitly in the excerpts; otherwise say it is not in the materials shown and the student should
check the full syllabus or instructor."""


def run_content_agent(question: str, sections: list[dict]) -> str:
    """Pass full outline + keyword-matched excerpts as hints; Ollama authors the answer."""
    outline = format_course_outline(sections)
    chunks = search_sections(question, sections, top_k=5)
    related = sections_context_for_llm(chunks)
    task = (
        "## Course map (from syllabus — full outline)\n"
        f"{outline}\n\n"
        "## Related syllabus excerpts (hints for this question)\n"
        f"{related}\n\n"
        "---\n\n"
        f"Student question:\n{question}"
    )
    return agent_run(ROLE_CONTENT_TUTOR, task, tools=None, model=MODEL, num_predict=1024)


# 3. Agent 2 — quizzes per section #############################################

ROLE_QUIZ_AUTHOR = """You write fair multiple-choice quizzes for one section of a course.
Return ONLY a single JSON object (no markdown fences, no commentary) with this shape:
{"section_id": "<copy exactly>", "title": "<section title>", "questions": [
  {"question": "...", "choices": ["A","B","C","D"], "correct_index": 0}
]}
Use exactly 3 questions. correct_index is 0-based. Vary difficulty slightly."""


def _extract_json_object(text: str) -> dict:
    """Pull the first JSON object from model output."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    blob = m.group(1) if m else None
    if not blob:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            blob = text[start : end + 1]
    if not blob:
        raise ValueError("Model did not return JSON.")
    return json.loads(blob)


def run_quiz_agent(sections: list[dict]) -> list[dict]:
    """
    Agent 2: For each parsed section, ask the LLM for a small MC quiz and collect results.

    Section boundaries come from markdown headings (deterministic split). The LLM only
    authors questions for each existing section.
    """
    quizzes: list[dict] = []
    for sec in sections:
        body = sec["body"] or ""
        if len(body) < 50:
            continue
        task = (
            f"Section id: {sec['id']}\n"
            f"Section title: {sec['title']}\n\n"
            f"Content:\n{body[:7000]}\n\n"
            f"Write the JSON quiz for this section only. section_id must be '{sec['id']}'."
        )
        raw = agent_run(ROLE_QUIZ_AUTHOR, task, tools=None, model=MODEL)
        try:
            quiz = _extract_json_object(raw)
        except (json.JSONDecodeError, ValueError):
            continue
        quiz["section_id"] = sec["id"]
        quiz["title"] = quiz.get("title") or sec["title"]
        qs = quiz.get("questions") or []
        if isinstance(qs, list) and len(qs) > 0:
            quizzes.append(quiz)
    return quizzes


# 4. Agent 3 — progress file via tool ##########################################

ROLE_PROGRESS = """You record course progress when requirements are met.
When the user message says the learner scored at least 80% on a section quiz,
you MUST call update_progress once with the provided section_id, score_percent,
and section_title. Do not invent scores."""


def update_progress(section_id: str, score_percent: float, section_title: str = "") -> dict:
    """
    Tool implementation: append a completion record if score_percent >= 80.

    This is the function Agent 3 invokes through Ollama tool calling.
    """
    if score_percent < 80:
        return {
            "ok": False,
            "message": "Scores below 80% are not written to progress.json.",
        }

    prog = load_progress()
    now = datetime.now(timezone.utc).isoformat()
    rows = prog.get("sections_completed") or []

    # Replace prior entry for the same section, keep newest pass
    rows = [r for r in rows if r.get("section_id") != section_id]
    rows.append(
        {
            "section_id": section_id,
            "section_title": section_title,
            "score_percent": float(score_percent),
            "completed_at": now,
        }
    )
    prog["sections_completed"] = rows
    prog["last_updated"] = now
    save_progress(prog)
    return {"ok": True, "recorded": {"section_id": section_id, "score_percent": score_percent}}


def _tool_update_metadata() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "update_progress",
                "description": (
                    "Write a quiz completion to progress.json when score_percent is 80 or higher."
                ),
                "parameters": {
                    "type": "object",
                    "required": ["section_id", "score_percent"],
                    "properties": {
                        "section_id": {"type": "string", "description": "Section id (e.g. sec-0)"},
                        "score_percent": {
                            "type": "number",
                            "description": "Percent correct on the quiz (0-100)",
                        },
                        "section_title": {
                            "type": "string",
                            "description": "Human-readable section title",
                        },
                    },
                },
            },
        }
    ]


def run_progress_agent(section_id: str, section_title: str, score_percent: float) -> str:
    """Agent 3: confirm via LLM + tool call; tool performs the JSON write."""
    tools = _tool_update_metadata()
    dispatch = {"update_progress": update_progress}
    user_msg = (
        f"The learner completed section '{section_title}' (id {section_id}) "
        f"with score_percent={score_percent:.1f}. "
        "If score_percent is 80 or higher, call update_progress with those exact values "
        "(include section_title). If below 80, respond with a short note and do not call the tool."
    )
    messages = [
        {"role": "system", "content": ROLE_PROGRESS},
        {"role": "user", "content": user_msg},
    ]
    return chat_with_tool_loop(messages, tools, dispatch, model=MODEL)


# 5. Orchestration helpers #####################################################


def load_sections_from_upload() -> list[dict]:
    """Parse the saved syllabus.md into sections for RAG and quizzes."""
    text = read_syllabus_text()
    if not text.strip():
        return []
    return parse_markdown_sections(text)


def regenerate_quizzes_for_upload() -> list[dict]:
    """Rebuild quizzes.json from the current syllabus (Agent 2)."""
    sections = load_sections_from_upload()
    quizzes = run_quiz_agent(sections)
    save_quizzes(quizzes)
    return quizzes


def grade_quiz(section_id: str, answers: list[int]) -> tuple[float, int, int]:
    """
    Score a submitted quiz. Returns (percent, correct_count, total).
    """
    quizzes = load_quizzes()
    quiz = next((q for q in quizzes if q.get("section_id") == section_id), None)
    if quiz is None:
        raise KeyError(f"No quiz for section_id={section_id}")
    questions = quiz.get("questions") or []
    n = len(questions)
    if n == 0:
        return 0.0, 0, 0
    if len(answers) != n:
        raise ValueError(f"Expected {n} answers, got {len(answers)}")
    correct = sum(
        1 for i, q in enumerate(questions) if int(answers[i]) == int(q.get("correct_index", -1))
    )
    pct = 100.0 * correct / n
    return pct, correct, n
