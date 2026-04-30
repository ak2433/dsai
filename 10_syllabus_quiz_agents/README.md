# Syllabus Quiz Agents

A demonstration app that combines **multi-agent orchestration**, **retrieval-augmented prompting** from a course markdown syllabus, and **Ollama function calling** for progress tracking. You can use the HTTP API or a terminal chat to ask a tutor questions, generate section quizzes, and record completions when scores are 80% or higher.

---

## Example syllabus file

**`History_of_Terrorism_Curriculum.md`** (in this folder) is a **sample course document** you can upload or load via `chat_cli.py`. It was produced primarily with **Claude (Anthropic)** as a drafting aid, then revised and enriched with the author’s **own domain knowledge**—so it reads like a plausible terrorism-studies-style curriculum but should be treated as **illustrative content** for testing the pipeline, not as an officially vetted academic program.

Any well-structured **`.md`** syllabus with headings works the same way; this file is optional convenience.

---

## System architecture: agent roles and workflow

The design uses **three logical agents** (separate system prompts and responsibilities in `agents.py`). They are composed into a **workflow** with clear stages: ingest syllabus → optional quiz generation → tutoring → quiz attempt → optional progress write.

### Agent roles

| Agent | Role | Inputs / outputs (conceptual) |
|-------|------|--------------------------------|
| **1 — Course tutor** | Explains topics in a way that **fits the course**: syllabus is **hints** (outline + related excerpts); Ollama may use **general knowledge** for teaching; **admin facts** (grades, due dates, required readings) must follow only what appears in the excerpts when stated as policy. | Input: student question + built prompt (course map + retrieved excerpts). Output: natural-language answer. **No tool calling** on this step (retrieval is done in Python). |
| **2 — Quiz author** | Writes **multiple-choice quizzes**, one Ollama call per substantial syllabus **section**. | Input: section id/title/body (body truncated in code). Output: JSON object parsed from model text; saved in `quizzes.json`. **No tool calling**. |
| **3 — Progress recorder** | When a quiz score is **≥ 80%**, drives the model to call a registered **tool** so completion is recorded—or the API **falls back** to the same Python function if the model omits a tool call. | Input: `section_id`, `section_title`, `score_percent`. Output: JSON from `update_progress`; `progress.json` updated on success. **Uses Ollama tool / function calling** via `chat_with_tool_loop`. |

### Workflow (end-to-end)

1. **Syllabus ingest** — Markdown is saved as `data/workspace/syllabus.md` (API `POST /upload` or `chat_cli.py` bootstrap).
2. **Parse** — `parse_markdown_sections` splits the file on `#`…`######` headings into `{id, title, body}` sections.
3. **Tutor path** — For each question: build **course map** (`format_course_outline`) + **keyword-ranked excerpts** (`search_sections` → `sections_context_for_llm`), then **one** Ollama chat (`agent_run`, no tools).
4. **Quiz path (optional)** — `POST /generate-quizzes` or CLI `generate quizzes`: loop sections → `agent_run` with quiz JSON schema → merge into `quizzes.json`.
5. **Attempt & progress** — `POST /submit-quiz` or CLI `quiz me`: score in Python; if **≥ 80%**, run `run_progress_agent` (tool loop) and optionally **fallback** `update_progress` in `app.py`.

```text
syllabus.md → parse sections
                 ├──→ [Agent 1] outline + search → Ollama (tutor reply)
                 └──→ [Agent 2] per-section body → Ollama → JSON quizzes
quiz submit (≥80%) → [Agent 3] Ollama + update_progress tool → progress.json
```

---

## RAG data source and search function

### Data source

- **Primary store:** `data/workspace/syllabus.md` — the last uploaded or copied syllabus (UTF-8 text).
- **Derived structure:** Sections are **not** stored in a separate DB; they are computed in memory from `syllabus.md` whenever needed (`parse_markdown_sections`).
- **Quiz artifacts:** `data/workspace/quizzes.json` (LLM-authored MCQs). **Progress:** `data/workspace/progress.json`.

### Search function (tutor “RAG”)

**Function:** `search_sections(query, sections, top_k=5)` in `syllabus_rag.py`.

- **Tokenization:** `query` is split into lowercase alphanumeric tokens (skips single-character noise); if that yields nothing, the whole trimmed string is used as one token.
- **Scoring:** For each section, score = count of query tokens that appear as **substrings** anywhere in `title + body` (case-insensitive). Sections are sorted by score descending; top **`top_k`** are kept.
- **Excerpt:** For each hit, up to **1200** characters of the section body are returned (with ellipsis if truncated), plus `id`, `title`, and internal `score`.
- **Empty query:** Returns the first sections with short excerpts and score `0.0` (lightweight default context).

**Companion helpers:**

- `format_course_outline(sections)` — every `sec-*` title for **global** course direction (always sent to the tutor).
- `sections_context_for_llm(chunks)` — formats search hits as markdown-style blocks for the prompt.

**Note:** Quiz authoring does **not** use this search function; it passes each section’s body directly to the quiz agent.

---

## Tool functions (Ollama function calling)

The app registers **one** tool for Ollama. Metadata is built in `_tool_update_metadata()`; the implementation is **`update_progress`** in `agents.py`; execution is routed by **`chat_with_tool_loop`**.

| Tool name | Purpose | Parameters | Returns |
|-----------|---------|------------|---------|
| **`update_progress`** | Append a **passed** section to `data/workspace/progress.json` when the learner scores **≥ 80%** on that section’s quiz (replace prior row for the same `section_id`). | **`section_id`** (string, required) — e.g. `sec-0`. **`score_percent`** (number, required) — 0–100. **`section_title`** (string, optional) — human-readable title. | **JSON-serializable dict:** On success: `{"ok": true, "recorded": {"section_id": "...", "score_percent": ...}}`. If `score_percent < 80`: `{"ok": false, "message": "Scores below 80% are not written to progress.json."}` (no file change). |

No other tools are exposed to Ollama in this project. The tutor and quiz author do not use the tools API.

---

## Technical details

### API keys and external services

- **No cloud LLM API keys** are required. The app talks to **Ollama** at **`http://localhost:11434`** (see `functions.py`: `OLLAMA_HOST`, `CHAT_URL`).
- **Optional:** Set environment variable **`OLLAMA_MODEL`** to override the default model name (default is in `functions.py`, e.g. `smollm2:1.7b`).

### HTTP API (FastAPI)

| Endpoint | Method | Summary |
|----------|--------|---------|
| `/health` | GET | Ollama reachability + workspace path |
| `/upload` | POST | Multipart `.md` → `syllabus.md` |
| `/sections` | GET | Parsed section ids/titles |
| `/ask` | POST | JSON `{"question": "..."}` → tutor answer |
| `/generate-quizzes` | POST | Build `quizzes.json` (slow) |
| `/quizzes` | GET | Questions + choices (no answers) |
| `/quizzes/{section_id}/answers` | GET | Full quiz + `correct_index` (instructor/debug) |
| `/submit-quiz` | POST | JSON `{"section_id", "answers": [0-based indices]}` → score + progress if ≥80% |
| `/progress` | GET | Current `progress.json` payload |

Interactive docs: **`http://127.0.0.1:8010/docs`** when the server is running (default port **8010** in examples).

### Python packages

From `requirements.txt`:

- `fastapi`, `uvicorn[standard]`, `python-multipart` — web app and file upload  
- `requests` — Ollama HTTP API  
- `pandas` — pulled in for shared `functions.py` helpers (`to_markdown`, etc.)

### Repository layout

```text
10_syllabus_quiz_agents/
├── app.py                 # FastAPI application
├── agents.py              # Agents 1–3, tool loop, grading
├── chat_cli.py            # Terminal REPL
├── functions.py           # Ollama agent / agent_run helpers
├── syllabus_rag.py        # Parse md, search_sections, outline
├── workspace.py           # Paths, JSON I/O for workspace files
├── requirements.txt
├── pyrightconfig.json     # Pylance include path
├── History_of_Terrorism_Curriculum.md  # Example syllabus (optional)
├── README.md
└── data/workspace/        # Created at runtime
    ├── syllabus.md
    ├── quizzes.json
    └── progress.json
```

---

## Usage instructions

### 1. Install dependencies

```bash
cd 10_syllabus_quiz_agents
pip install -r requirements.txt
```

Use a virtual environment if you prefer (`python -m venv .venv` then activate).

### 2. Install and run Ollama

1. Install [Ollama](https://ollama.com/) and start it (so **`http://localhost:11434`** responds).  
2. Pull a model your machine can run, matching or overriding the default in `functions.py`, for example:

   ```bash
   ollama pull smollm2:1.7b
   ```

3. **Optional:** set **`OLLAMA_MODEL`** to another pulled model name before starting the app or CLI.

There are **no API keys** in `.env` for this project unless you add your own integrations.

### 3. Prepare the syllabus (data source)

- **HTTP:** `POST /upload` with a `.md` file (max ~2 MB).  
- **CLI:** Run `chat_cli.py`; if `data/workspace/syllabus.md` is empty, it can offer **`History_of_Terrorism_Curriculum.md`** or a path you type.  
- **Manual:** Copy a markdown file to **`data/workspace/syllabus.md`**.

### 4. Run the system

**Web API (recommended for Swagger testing):**

```bash
uvicorn app:app --reload --port 8010
```

Then open **`http://127.0.0.1:8010/docs`**. Typical order: **upload** → **generate-quizzes** → **ask** / **quizzes** / **submit-quiz** → **progress**.

**Terminal only:**

```bash
python chat_cli.py
```

Commands include normal chat, **`quiz me`**, **`generate quizzes`**, **`progress`**, **`help`**, **`quit`**.

### 5. Configure nothing else for a default demo

- **Port:** change `8010` in the `uvicorn` command if needed.  
- **Ollama host:** default is localhost; only change if you run Ollama on another machine (would require editing `functions.py` or adding env support there).

---

## Author

Tony Klar.
