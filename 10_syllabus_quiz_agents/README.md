# Syllabus Quiz Agents

A small demonstration project that combines **multi-agent orchestration**, **retrieval-augmented style context** from a course markdown file, and **Ollama function calling**. You upload a syllabus (or load one locally), chat with a tutor, generate section quizzes, and record progress when quiz scores reach 80% or higher.

## What it demonstrates (course concepts)

### Multi-agent workflow

Three **distinct agent roles** are implemented in `agents.py`. They are not one monolithic prompt; each has its own system instructions and responsibilities.

| Agent | Role | How it runs |
|-------|------|-------------|
| **1 — Course tutor** | Answers student questions in a way that fits **your** course | Retrieves syllabus context in Python, then one Ollama chat via `agent_run` (`functions.py`) with **no** tools on that call. |
| **2 — Quiz author** | Builds multiple-choice quizzes **per syllabus section** | Markdown is split on headings in `syllabus_rag.py`; for each section with enough text, Ollama is called once to emit structured JSON. Orchestration is a **loop over sections**, not a single mega-prompt. |
| **3 — Progress recorder** | Decides when to persist a passing quiz to disk | Ollama is given a **tool definition** and must call `update_progress` when the scenario qualifies. |

Together this is a **pipeline**: tutor ↔ separate quiz generation path ↔ post-quiz progress step. That mirrors lab-style “multiple agents / multiple steps” even though each step is implemented as one or more LLM calls with clear handoffs (context building → LLM → parsing or tools).

### RAG (retrieval + generation)

“RAG” here means **retrieval happens outside the model**, then the **retrieved text is injected** into the prompt.

- **Ingestion:** The syllabus is stored as `data/workspace/syllabus.md`. It is parsed into sections by markdown headings (`parse_markdown_sections` in `syllabus_rag.py`).
- **Retrieval:** For the tutor, `search_sections` scores sections with simple **keyword overlap** between the student’s question and each section’s title + body. The top matches are formatted as excerpts (`sections_context_for_llm`).
- **Course map:** The **full outline** of all section titles is also passed in every tutor turn (`format_course_outline`) so the model sees how the course is **directed** end-to-end, not only the top-matched chunks.
- **Generation:** Ollama produces the visible answer. The design treats excerpts as **hints and scope** (vocabulary, emphasis, organization); the model may use **general knowledge** to teach, while **binding admin** (grades, due dates, required readings) is instructed to follow **only** what appears in the excerpts.

So this is **retrieval-augmented prompting**: retrieve → augment the message → generate. It is **not** strict “answer only from snippets” RAG for conceptual questions; it is **course-aware tutoring** with grounded admin rules.

Quiz authoring (Agent 2) is **not** keyword RAG; each section’s **full local text** (up to a length cap) is passed directly into the quiz prompt.

### Tool / function calling

Ollama’s chat API supports **tools**: JSON metadata describing a function name, description, and parameters. When the model returns `tool_calls`, your code must **execute** the matching Python function and send results back in a follow-up turn.

- **Helper loop:** `chat_with_tool_loop` in `agents.py` posts to Ollama with `tools`, reads `tool_calls`, runs the mapped Python callables, appends `role: "tool"` messages, and repeats until the model returns normal text.
- **Tool used in this project:** `update_progress(section_id, score_percent, section_title)` — implemented in `agents.py`, exposed to Ollama via `_tool_update_metadata()`. When a quiz is submitted and the score is **≥ 80%**, **`run_progress_agent`** drives the model to call this tool so **Agent 3** records completion in `data/workspace/progress.json`.
- **Reliability note:** The FastAPI route in `app.py` can **fallback** to calling `update_progress` in Python if the model never emits a tool call, so progress is still saved for demos.

The **tutor (Agent 1) does not use tools** on purpose: small local models often emit fake tool markup in plain text instead of real `tool_calls`. Retrieval is done in Python; the tutor call is a single completion.

Shared HTTP helpers and `agent` / `agent_run` live in **`functions.py`** in this folder (Ollama URL, timeouts, optional `num_predict`).

## Requirements

- Python 3.10+ recommended  
- [Ollama](https://ollama.com/) running locally with a suitable model pulled (see your course default in `functions.py`, or set **`OLLAMA_MODEL`**)

## Setup

```bash
cd 10_syllabus_quiz_agents
pip install -r requirements.txt
```

## Run the HTTP API

```bash
uvicorn app:app --reload --port 8010
```

Open [http://127.0.0.1:8010/docs](http://127.0.0.1:8010/docs) for interactive Swagger.

Typical flow: **`POST /upload`** (`.md` syllabus) → **`POST /generate-quizzes`** → **`POST /ask`** for questions → **`GET /quizzes`** / **`POST /submit-quiz`** for attempts → **`GET /progress`**.

## Run the terminal chat (no browser)

```bash
python chat_cli.py
```

Type normal questions for the tutor, **`quiz me`** for an interactive quiz, **`generate quizzes`** to rebuild quizzes, **`progress`** to print JSON state, **`quit`** to exit.

## Data layout

All mutable state lives under **`data/workspace/`**:

| File | Purpose |
|------|---------|
| `syllabus.md` | Last uploaded syllabus |
| `quizzes.json` | MCQ sets keyed by `section_id` |
| `progress.json` | Sections completed at ≥ 80% (and timestamps) |

## Project files (quick map)

| File | Purpose |
|------|---------|
| `app.py` | FastAPI routes |
| `agents.py` | Three agents, `chat_with_tool_loop`, grading, progress tooling |
| `syllabus_rag.py` | Parse markdown sections, keyword search, outline + excerpt formatting |
| `workspace.py` | Paths and JSON I/O |
| `functions.py` | Ollama `agent` / `agent_run` helpers |
| `chat_cli.py` | Terminal REPL for testing |

## Author

Tim Fraser (course materials).
