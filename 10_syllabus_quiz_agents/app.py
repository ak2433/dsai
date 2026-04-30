# app.py
# FastAPI app: upload syllabus, ask the tutor, build quizzes, submit attempts
# Tim Fraser

# Run from this folder:
#   pip install -r requirements.txt
#   uvicorn app:app --reload --port 8010
#
# Terminal testing (no HTTP): python chat_cli.py

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from functions import ensure_ollama_available
from agents import (
    grade_quiz,
    load_sections_from_upload,
    regenerate_quizzes_for_upload,
    run_content_agent,
    run_progress_agent,
    update_progress,
)
from workspace import (
    WORKSPACE_DIR,
    ensure_workspace,
    load_progress,
    load_quizzes,
    write_syllabus_bytes,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: same as former on_event("startup")
    ensure_workspace()
    try:
        ensure_ollama_available()
    except RuntimeError:
        # Let endpoints surface a clear error when the model is actually called
        pass
    yield


app = FastAPI(title="Syllabus quiz agents", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    try:
        ensure_ollama_available()
        ollama_ok = True
    except RuntimeError:
        ollama_ok = False
    ensure_workspace()
    return {"ollama": ollama_ok, "workspace": str(WORKSPACE_DIR)}


class AskBody(BaseModel):
    question: str = Field(..., min_length=1, description="Student question about course content")


@app.post("/ask")
def ask_course_agent(body: AskBody) -> dict:
    sections = load_sections_from_upload()
    if not sections:
        raise HTTPException(status_code=400, detail="Upload a syllabus markdown file first (/upload).")
    answer = run_content_agent(body.question, sections)
    return {"answer": answer}


class SubmitQuizBody(BaseModel):
    section_id: str = Field(..., description="Section id such as sec-0")
    answers: list[int] = Field(..., description="Selected choice index per question, same order as quiz")


@app.post("/submit-quiz")
def submit_quiz(body: SubmitQuizBody) -> dict:
    try:
        pct, correct, n = grade_quiz(body.section_id, body.answers)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    sections = load_sections_from_upload()
    title = ""
    for s in sections:
        if s["id"] == body.section_id:
            title = s["title"]
            break

    agent_note = ""
    if pct >= 80:
        agent_note = run_progress_agent(body.section_id, title, pct)
        # If the model skipped the tool, still persist progress (deterministic path).
        prog_now = load_progress()
        done_ids = {r.get("section_id") for r in (prog_now.get("sections_completed") or [])}
        if body.section_id not in done_ids:
            update_progress(body.section_id, pct, title)
            agent_note += "\n(Fallback: progress written in Python because the model did not call the tool.)"
    else:
        agent_note = "Score under 80% — progress file not updated."

    return {
        "score_percent": round(pct, 2),
        "correct": correct,
        "total": n,
        "progress_updated": pct >= 80,
        "agent_progress_output": agent_note,
    }


@app.post("/upload")
async def upload_syllabus(file: UploadFile = File(...)) -> dict:
    if not file.filename or not file.filename.lower().endswith(".md"):
        raise HTTPException(status_code=400, detail="Please upload a .md (markdown) file.")
    raw = await file.read()
    if len(raw) > 2_000_000:
        raise HTTPException(status_code=400, detail="File too large (max ~2 MB).")
    write_syllabus_bytes(raw)
    sections = load_sections_from_upload()
    return {"filename": file.filename, "sections_found": len(sections), "section_titles": [s["title"] for s in sections]}


@app.post("/generate-quizzes")
def generate_quizzes() -> dict:
    sections = load_sections_from_upload()
    if not sections:
        raise HTTPException(status_code=400, detail="Upload a syllabus first.")
    quizzes = regenerate_quizzes_for_upload()
    return {"quizzes_built": len(quizzes), "section_ids": [q["section_id"] for q in quizzes]}


@app.get("/quizzes")
def list_quizzes() -> dict:
    """Return quiz stems without revealing correct_index (for a simple front-end)."""
    out = []
    for q in load_quizzes():
        stripped = {
            "section_id": q.get("section_id"),
            "title": q.get("title"),
            "questions": [
                {"question": qq.get("question"), "choices": qq.get("choices")} for qq in (q.get("questions") or [])
            ],
        }
        out.append(stripped)
    return {"quizzes": out}


@app.get("/quizzes/{section_id}/answers")
def teacher_view_answers(section_id: str) -> dict:
    """Full quiz including correct_index (useful for debugging or instructor mode)."""
    for q in load_quizzes():
        if q.get("section_id") == section_id:
            return {"quiz": q}
    raise HTTPException(status_code=404, detail="Unknown section_id")


@app.get("/progress")
def progress() -> dict:
    return load_progress()


@app.get("/sections")
def sections_outline() -> dict:
    sections = load_sections_from_upload()
    return {"sections": [{"id": s["id"], "title": s["title"]} for s in sections]}
