# chat_cli.py
# Terminal chat to test syllabus Q&A and quizzes (no browser)
# Tim Fraser

# Run from this folder:
#   python chat_cli.py
#
# Ask the course tutor anything; type "quiz me" for an interactive multiple-choice quiz.
# Other commands: help, progress, generate quizzes, quit

import sys
from pathlib import Path

from agents import (
    grade_quiz,
    load_sections_from_upload,
    regenerate_quizzes_for_upload,
    run_content_agent,
    run_progress_agent,
    update_progress,
)
from workspace import ensure_workspace, load_progress, load_quizzes, read_syllabus_text, write_syllabus_bytes

# Phrases that start quiz mode (normalized: lower strip)
QUIZ_TRIGGERS = frozenset({"quiz me", "quiz", "test me"})


def _print_help() -> None:
    print(
        """
Commands (any other line is sent to the course tutor):
  quiz me          Take a multiple-choice quiz on one section
  progress         Show progress.json (sections passed at 80%+ )
  generate quizzes Rebuild quizzes from the syllabus (slow; uses Ollama)
  help             This message
  quit / exit      Leave the chat
"""
    )


def _bootstrap_syllabus_if_empty() -> bool:
    """If workspace has no syllabus, offer the repo sample file or a path you type."""
    if read_syllabus_text().strip():
        return True

    here = Path(__file__).resolve().parent
    sample = here / "History_of_Terrorism_Curriculum.md"
    print("No syllabus found in data/workspace/syllabus.md.")
    if sample.is_file():
        use = input(f"Load sample {sample.name}? [Y/n]: ").strip().lower()
        if use in ("", "y", "yes"):
            write_syllabus_bytes(sample.read_bytes())
            print("Loaded.")
            return True

    path = input("Or enter path to a .md syllabus file (Enter to skip): ").strip().strip('"')
    if path:
        p = Path(path).expanduser()
        if not p.is_file():
            print(f"Not found: {p}")
            return False
        write_syllabus_bytes(p.read_bytes())
        print("Loaded.")
        return True

    print("You can still run 'generate quizzes' after adding a syllabus via the API /upload.")
    return False


def _record_progress_if_passed(section_id: str, title: str, pct: float) -> str:
    """Same logic as app.py: agent + fallback write."""
    if pct < 80:
        return "Score under 80% — progress file not updated."
    note = run_progress_agent(section_id, title, pct)
    prog_now = load_progress()
    done_ids = {r.get("section_id") for r in (prog_now.get("sections_completed") or [])}
    if section_id not in done_ids:
        update_progress(section_id, pct, title)
        note += "\n(Fallback: progress written in Python.)"
    return note


def _pick_quiz() -> dict | None:
    quizzes = load_quizzes()
    if not quizzes:
        return None
    print("\nAvailable quizzes:")
    for i, q in enumerate(quizzes):
        print(f"  [{i}] {q.get('section_id')} — {q.get('title')}")
    raw = input("Pick a number (or Enter for 0): ").strip()
    idx = int(raw) if raw else 0
    if idx < 0 or idx >= len(quizzes):
        print("Invalid choice.")
        return None
    return quizzes[idx]


def _run_quiz_session() -> None:
    quizzes = load_quizzes()
    if not quizzes:
        gen = input("No quizzes yet. Generate them now? This calls Ollama and may take a while [y/N]: ")
        if gen.strip().lower() not in ("y", "yes"):
            return
        print("Generating…")
        sections = load_sections_from_upload()
        if not sections:
            print("No syllabus text in workspace. Add syllabus.md first.")
            return
        regenerate_quizzes_for_upload()
        quizzes = load_quizzes()
        if not quizzes:
            print("Quiz generation produced no quizzes (check Ollama / model).")
            return

    quiz = _pick_quiz()
    if not quiz:
        return

    section_id = quiz.get("section_id", "")
    title = quiz.get("title", "")
    questions = quiz.get("questions") or []
    if not questions:
        print("That quiz has no questions.")
        return

    print(f"\n--- Quiz: {title} ({section_id}) — enter 1–n for each question ---\n")
    answers: list[int] = []
    for j, qq in enumerate(questions, start=1):
        print(f"Q{j}. {qq.get('question')}")
        choices = qq.get("choices") or []
        for k, c in enumerate(choices):
            print(f"   {k + 1}. {c}")
        while True:
            line = input("Your choice (number): ").strip()
            try:
                choice_num = int(line)
            except ValueError:
                print("Enter a number.")
                continue
            if 1 <= choice_num <= len(choices):
                answers.append(choice_num - 1)
                break
            print(f"Pick 1–{len(choices)}.")

    try:
        pct, correct, n = grade_quiz(section_id, answers)
    except (KeyError, ValueError) as e:
        print(f"Grading error: {e}")
        return

    print(f"\nScore: {correct}/{n} = {pct:.1f}%")
    note = _record_progress_if_passed(section_id, title, pct)
    print(note)


def main() -> None:
    ensure_workspace()
    print("Syllabus quiz — terminal chat (Ctrl+C to exit)\n")
    if not _bootstrap_syllabus_if_empty():
        print("Tip: place markdown in data/workspace/syllabus.md or start FastAPI and POST /upload.\n")

    sections = load_sections_from_upload()
    if sections:
        print(f"Syllabus loaded: {len(sections)} section(s). Type 'help' for commands.\n")
    else:
        print("No sections parsed yet — add a syllabus before asking questions.\n")

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            sys.exit(0)

        if not line:
            continue

        low = line.lower()
        if low in ("quit", "exit", "q"):
            print("Bye.")
            sys.exit(0)
        if low == "help":
            _print_help()
            continue
        if low == "progress":
            print(load_progress())
            continue
        if low in ("generate quizzes", "gen quizzes", "regenerate quizzes"):
            if not load_sections_from_upload():
                print("Need a syllabus first.")
                continue
            print("Generating quizzes (Ollama)…")
            regenerate_quizzes_for_upload()
            print(f"Done. {len(load_quizzes())} quiz(es) in data/workspace/quizzes.json")
            continue
        if low in QUIZ_TRIGGERS:
            _run_quiz_session()
            continue

        # Tutor turn
        sections = load_sections_from_upload()
        if not sections:
            print("No syllabus in workspace — cannot answer.")
            continue
        try:
            answer = run_content_agent(line, sections)
            print(f"tutor> {answer}\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
