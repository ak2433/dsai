# assigner_staff_clients.py
# ACTIVITY: Staff–client assignment (see ACTIVITY_assigner.md)
# pip install requests python-dotenv
#
# Stage 1: assignment table + summary. Stage 2: stress-test follow-up (same chat thread).
# Override the follow-up with env ASSIGNER_STRESS_TEST (one line) or edit DEFAULT_STAGE2_USER below.

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

DEFAULT_OLLAMA_MODEL = "smollm2:1.7b"

SYSTEM_PROMPT = """You are a managing partner at a consulting firm making staffing assignments.
Your job is to read unstructured descriptions of staff members and clients,
then assign each staff member to exactly 2 clients based on fit.

Return:
1. An assignment table with columns: Staff Member | Client 1 | Client 2 | Rationale (1 sentence)
2. A brief paragraph (3–5 sentences) summarizing your overall assignment logic

Rules:
- Each staff member gets exactly 2 clients
- Each client is assigned to exactly 1 staff member
- No client may be left unassigned
- Base assignments on demonstrated fit — skills, experience, communication style
- Flag any assignments where fit is weak and explain why"""

STAFF_CLIENT_DATA = """
--- STAFF ---

Alex Chen
Senior consultant, 9 years experience. Background in financial services and
regulatory compliance. Known for being methodical and detail-oriented.
Prefers clients who are organized and have clear deliverables.
Not great with ambiguous or fast-moving projects.

Brianna Okafor
Mid-level consultant, 4 years experience. Specialist in nonprofit and public
sector work. Very strong communicator — clients love her. Comfortable with
messy, evolving scopes. Has done a lot of stakeholder engagement work.

Carla Mendez
Senior consultant, 7 years experience. Deep expertise in healthcare and life
sciences. Data-heavy work is her strength — she's built several dashboards and
automated reporting tools. Tends to be blunt and efficient; not the warmest
bedside manner but clients respect her results.

Dana Park
Junior consultant, 2 years experience. Background is in marketing and consumer
research. Eager and creative. Better on smaller, well-defined tasks.
Still building confidence with senior client stakeholders.

Elliot Vasquez
Partner-level, 15 years experience. Generalist with a strong track record in
strategy and organizational change. Good relationship manager. Prefers high-stakes,
high-visibility engagements. Gets bored on smaller tactical work.

Fiona Marsh
Mid-level consultant, 5 years experience. Former journalist turned researcher.
Excellent writer and communicator. Often assigned to deliverable-heavy projects
(reports, white papers, presentations). Works well independently.
Prefers clients who give her creative latitude.

--- CLIENTS ---

Client A — Riverdale Community Health Clinic
Small nonprofit health clinic undergoing a strategic planning process.
Moderate budget. Stakeholders include the board, medical staff, and community
advocates. Very collaborative, but decisions are slow due to committee structure.
Main need: facilitation support and a written strategic plan.

Client B — Atlas Financial Group
Large regional bank. Highly regulated environment. Project involves auditing
their compliance documentation and recommending process improvements.
Very organized client — they have a detailed project plan. Expects formal
deliverables and regular status reports.

Client C — BrightPath Schools (Charter Network)
Fast-growing charter school network. Expanding from 3 to 8 schools.
Needs help with org design and HR policy. Client is enthusiastic but somewhat
disorganized. Decision-maker is the founder/CEO — she's visionary but hard to pin
down for meetings.

Client D — Nexagen Pharmaceuticals
Mid-size pharma company. Project is a data audit and KPI dashboard buildout
for their clinical operations team. Technical stakeholders who want results,
not hand-holding. Timeline is tight.

Client E — Greenway Transit Authority
Regional transit agency. Unionized workforce. Project involves a service
redesign study with significant community engagement components.
Political sensitivities — several board members have conflicting opinions.
Long timeline, phased project.

Client F — Solstice Consumer Goods
Consumer packaged goods brand. Needs a market research summary and brand
positioning analysis ahead of a product launch. Fun client, collaborative,
lots of back and forth. Not a huge budget. Creative work valued.

Client G — Meridian Capital Partners
Private equity firm. Fast-moving, high-expectations. Needs an org assessment
of a portfolio company. Very low patience for process — they want findings fast.
Elliot has a pre-existing relationship with the managing partner.

Client H — Harbor City Government (Parks Dept.)
Municipal parks department doing a 10-year capital planning study.
Lots of stakeholders — parks staff, city council, community groups.
Needs public engagement support and a formal report for the city council.

Client I — ClearView Diagnostics
Healthcare tech startup. Building a clinical decision support tool.
Needs help structuring their regulatory strategy and drafting FDA submission
materials. Technical and regulatory complexity is high. Startup culture —
informal, fast, sometimes chaotic.

Client J — The Holloway Foundation
Private philanthropy. Wants a landscape scan and strategic options memo on
workforce development funding. Small team, thoughtful, low-maintenance.
Primarily needs a polished, well-written deliverable.

Client K — Summit Retail Group
Multi-location retail chain. Undergoing a cost reduction initiative.
Wants operational benchmarking and process recommendations.
Client stakeholders are skeptical of consultants — they've had bad experiences
before. Need someone who can build trust quickly.

Client L — Vance Biomedical Research Institute
Academic research institute. Needs help redesigning their grant reporting
process and building a data tracking system. Methodical, detail-oriented
stakeholders. Comfortable with technical complexity.
""".strip()

USER_PROMPT_STAGE1 = (
    "Below are descriptions of our 6 staff members and 12 clients.\n"
    "Please make the best possible assignments.\n\n"
    + STAFF_CLIENT_DATA
)

# Default stress-test (edit after Stage 1 if you want a different pairing). Optional: ASSIGNER_STRESS_TEST in .env.
DEFAULT_STAGE2_USER = (
    "I'm not sure about the assignment of Dana Park to Meridian Capital Partners.\n"
    "Can you reconsider this pairing and either defend it or suggest an alternative?"
)


def _ollama_chat(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    api_key: str | None,
    timeout_s: int = 600,
) -> str:
    url = base_url.rstrip("/") + "/api/chat"
    body: dict = {"model": model, "messages": messages, "stream": False}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    r = requests.post(url, json=body, headers=headers, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    return str(data.get("message", {}).get("content", ""))


def main() -> None:
    here = Path(__file__).resolve().parent
    os.chdir(here)
    load_dotenv(here / ".env")
    if not (os.getenv("OLLAMA_MODEL") or "").strip():
        os.environ["OLLAMA_MODEL"] = DEFAULT_OLLAMA_MODEL

    print("\n📋 Staff–client assigner — ACTIVITY_assigner.md\n")

    base_url = (os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
    model = (os.getenv("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL).strip()
    print(f"🔧 OLLAMA_MODEL={model}\n")

    api_key = (os.getenv("OLLAMA_API_KEY") or "").strip() or None
    if api_key and "ollama.com" not in base_url:
        print("⚠️  OLLAMA_API_KEY is set but host is not ollama.com — sending Bearer anyway.\n")
    if not api_key and "ollama.com" in base_url:
        print("❌ Ollama Cloud requires OLLAMA_API_KEY in .env", file=sys.stderr)
        sys.exit(1)

    stress_user = (os.getenv("ASSIGNER_STRESS_TEST") or "").strip() or DEFAULT_STAGE2_USER

    out1 = here / "output" / "assigner_stage1.md"
    out2 = here / "output" / "assigner_stage2.md"

    base_messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_STAGE1},
    ]

    print(f"\n{'=' * 60}")
    print("☁️  Stage 1 — initial assignments")
    print(f"   Model: {model}")
    print(f"{'=' * 60}\n")
    t0 = time.perf_counter()
    reply1 = _ollama_chat(
        base_url=base_url, model=model, messages=base_messages, api_key=api_key
    )
    t1 = time.perf_counter()
    out1.parent.mkdir(parents=True, exist_ok=True)
    out1.write_text(
        f"<!-- Stage 1 | model={model} | seconds={t1 - t0:.1f} -->\n\n{reply1}",
        encoding="utf-8",
    )
    print(f"✅ Wrote {out1} ({len(reply1):,} chars) in {t1 - t0:.1f}s\n")

    messages_stress = [
        *base_messages,
        {"role": "assistant", "content": reply1},
        {"role": "user", "content": stress_user},
    ]

    print(f"\n{'=' * 60}")
    print("☁️  Stage 2 — stress-test follow-up (multi-turn)")
    print(f"{'=' * 60}\n")
    t0 = time.perf_counter()
    reply2 = _ollama_chat(
        base_url=base_url, model=model, messages=messages_stress, api_key=api_key
    )
    t1 = time.perf_counter()

    stage2_doc = (
        f"<!-- Stage 2 | model={model} | seconds={t1 - t0:.1f} -->\n\n"
        "### Follow-up (user)\n\n"
        f"{stress_user}\n\n"
        "### Model response\n\n"
        f"{reply2}"
    )
    out2.write_text(stage2_doc, encoding="utf-8")
    print(f"✅ Wrote {out2} ({len(reply2):,} chars) in {t1 - t0:.1f}s\n")

    print("─" * 60)
    print("💾 Artifacts")
    print(f"   📄 {out1}")
    print(f"   📄 {out2}")
    print("   📝 REFLECTION_assigner.md — Stage 3 submission text")
    print("─" * 60 + "\n")


if __name__ == "__main__":
    main()
