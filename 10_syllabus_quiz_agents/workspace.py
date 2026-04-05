# workspace.py
# Paths and JSON state for uploaded syllabus, quizzes, and progress
# Tim Fraser

# Keeps all learner state under data/workspace/ so the FastAPI app stays stateless
# across restarts except for files on disk.

import json
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent / "data" / "workspace"
SYLLABUS_PATH = WORKSPACE_DIR / "syllabus.md"
QUIZZES_PATH = WORKSPACE_DIR / "quizzes.json"
PROGRESS_PATH = WORKSPACE_DIR / "progress.json"


def ensure_workspace() -> None:
    """Create data directories if missing."""
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)


def read_syllabus_text() -> str:
    """Return uploaded syllabus markdown, or empty string if none."""
    if not SYLLABUS_PATH.is_file():
        return ""
    return SYLLABUS_PATH.read_text(encoding="utf-8", errors="replace")


def write_syllabus_bytes(data: bytes) -> None:
    """Save uploaded file as syllabus.md."""
    ensure_workspace()
    SYLLABUS_PATH.write_bytes(data)


def load_quizzes() -> list[dict]:
    if not QUIZZES_PATH.is_file():
        return []
    return json.loads(QUIZZES_PATH.read_text(encoding="utf-8"))


def save_quizzes(quizzes: list[dict]) -> None:
    ensure_workspace()
    QUIZZES_PATH.write_text(json.dumps(quizzes, indent=2), encoding="utf-8")


def load_progress() -> dict:
    if not PROGRESS_PATH.is_file():
        return {
            "syllabus_path": str(SYLLABUS_PATH.name),
            "sections_completed": [],
            "last_updated": None,
        }
    return json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))


def save_progress(data: dict) -> None:
    ensure_workspace()
    PROGRESS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
