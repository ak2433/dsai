# decider_venues.py
# ACTIVITY: Wedding venue comparison (see ACTIVITY_decider.md)
# pip install requests python-dotenv
#
# Local Ollama (default): OLLAMA_HOST=http://127.0.0.1:11434; model below unless OLLAMA_MODEL is set.
# Ollama Cloud: set OLLAMA_HOST=https://ollama.com, OLLAMA_MODEL=..., OLLAMA_API_KEY=...

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# Default model (matches 06_agents/02_using_ollama.py). Override via OLLAMA_MODEL in .env or environment.
DEFAULT_OLLAMA_MODEL = "smollm2:1.7b"

# ---------------------------------------------------------------------------
# Embedded course content (same as ACTIVITY_decider.md)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a structured data extractor and decision analyst.
Your job is to extract key attributes from unstructured venue descriptions,
build a comparison table, and recommend the top 3 venues based on the client's priorities.

Always return:
1. A markdown table with columns: Venue, Capacity, Approx. Price/Night, Catering, Outdoor, Parking, Vibe (1 word)
2. A ranked shortlist of top 3 venues with 1-sentence justification each
3. One sentence noting any venues you had to exclude due to missing information

Be concise. Do not invent data that is not in the descriptions."""

VENUE_DATA = """
Venue 1 — The Rosewood Estate
A sprawling property in the Hudson Valley with manicured gardens and a restored barn.
Capacity up to 175 guests. Rental fee is $17,500 Friday–Sunday. They have a preferred
catering list with 4 approved vendors. Outdoor ceremony space available with a rain
backup tent. Parking for ~80 cars on site.

Venue 2 — The Grand Metropolitan Hotel
Downtown ballroom, seats up to 300. In-house catering only. Pricing starts at $12,000
for the ballroom rental, catering packages extra. Valet parking. No outdoor space.

Venue 3 — Lakeview Pavilion
Outdoor lakeside pavilion. No indoor backup. BYOB catering. Fits about 90 people
comfortably, 110 at a squeeze. Very affordable — around $2,500 for a weekend.

Venue 4 — Thornfield Manor
Historic manor house, 8 acres. Exclusive use for the weekend. Price: $18,000.
In-house catering team. Ceremony can be held on the grounds or in the chapel.
Capacity 150. Featured in several bridal magazines.

Venue 5 — The Foundry at Millworks
Industrial-chic converted factory. Very trendy. Capacity 250. Bring your own vendors.
Rental is $5,000. Rooftop available for cocktail hour. No on-site parking — street
parking and nearby garage only.

Venue 6 — Sunrise Farm & Vineyard
Working vineyard with barn and outdoor ceremony terrace. Stunning views. Capacity 130.
Weekend rental $9,800. Catering through their in-house team or 2 approved vendors.
Ample parking. Very popular — books 18 months out.

Venue 7 — The Atrium Club
Corporate event space that does weddings on weekends. Very flexible on catering.
Fits 300+. Located downtown. Pricing on request — sales team says "typically $9,000–$14,000
depending on date." Not particularly romantic but very professional.

Venue 8 — Cedar Hollow Retreat
Rustic woodland lodge. Intimate and cozy. Max 60 guests. $3,200 for a Saturday.
Outside catering allowed. No formal parking lot — guests park in a field.

Venue 9 — The Belvedere
Upscale rooftop venue with skyline views. Indoor/outdoor setup. Capacity 180.
In-house catering required. Rental + minimum catering spend is $28,000.
Very elegant. Valet only.

Venue 10 — Harborside Event Center
Waterfront venue, brand new. Capacity 220. Pricing TBD — still finalizing packages.
Flexible on catering. Outdoor terrace available. Large parking lot.

Venue 11 — The Ivy House
Garden venue in a residential neighborhood. Permits outdoor ceremonies.
Capacity 100. $4,500 rental. BYOB catering. Street parking only — coordinator
recommends a shuttle from a nearby lot.

Venue 12 — Maple Ridge Country Club
Classic country club setting. Capacity 160. In-house catering only, known for
being very good. Rental from $28,500. Golf course backdrop for photos.
Ample parking. Private feel.

Venue 13 — The Glasshouse Conservatory
All-glass event space surrounded by botanical gardens. Very dramatic.
Capacity 140. $18,000 rental, catering open. Outdoor garden available for ceremonies.
Parking on site. Popular for spring weddings.

Venue 14 — Millbrook Inn
Country inn with event lawn. Venue rental $10,500. Capacity 120. Outside catering
allowed. Some overnight rooms available for wedding party. Very charming.

Venue 15 — The Warehouse District Loft
Raw, urban space. Very minimal. No catering kitchen. Capacity 200.
$8,800 rental. Not ideal for traditional weddings.

Venue 16 — Cloverfield Farms
Family-owned working farm. Barn + outdoor space. Capacity 135.
$6,000 Friday–Sunday. Preferred caterer list (3 vendors).
Casual, warm atmosphere. Lots of parking. Dogs welcome.
""".strip()

PRIORITIES_STAGE1 = """Here are the couple's priorities:
- Budget: under $8,000 for venue rental
- Guest count: ~120 people
- Vibe: romantic, not too corporate
- Must have outdoor ceremony option
- Catering must be in-house or on an approved vendor list

Here are descriptions of 16 venues. Please analyze and recommend.

""" + VENUE_DATA

PRIORITIES_STAGE2 = """Here are the couple's priorities:
- Budget: flexible, up to $15,000
- Guest count: ~200 people
- Vibe: elegant, grand
- Outdoor is a nice-to-have but not required
- No catering constraint

Here are descriptions of 16 venues. Please analyze and recommend.

""" + VENUE_DATA


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


def _run_stage(
    label: str,
    user_content: str,
    *,
    base_url: str,
    model: str,
    api_key: str | None,
    out_path: Path,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"☁️  {label}")
    print(f"   Model: {model}")
    print(f"   POST {base_url.rstrip('/')}/api/chat")
    print(f"{'=' * 60}\n")
    t0 = time.perf_counter()
    reply = _ollama_chat(
        base_url=base_url,
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        api_key=api_key,
    )
    elapsed = time.perf_counter() - t0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = f"<!-- {label} | model={model} | seconds={elapsed:.1f} -->\n\n"
    out_path.write_text(header + reply, encoding="utf-8")
    print(f"✅ Wrote {out_path} ({len(reply):,} chars) in {elapsed:.1f}s\n")


def main() -> None:
    here = Path(__file__).resolve().parent
    os.chdir(here)
    load_dotenv(here / ".env")
    if not (os.getenv("OLLAMA_MODEL") or "").strip():
        os.environ["OLLAMA_MODEL"] = DEFAULT_OLLAMA_MODEL

    print("\n📋 Wedding venue decider — ACTIVITY_decider.md\n")

    base_url = (os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
    model = (os.getenv("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL).strip()
    print(f"🔧 OLLAMA_MODEL={model}\n")

    api_key = (os.getenv("OLLAMA_API_KEY") or "").strip() or None
    if api_key and "ollama.com" not in base_url:
        print("⚠️  OLLAMA_API_KEY is set but host is not ollama.com — sending Bearer anyway.\n")
    if not api_key and "ollama.com" in base_url:
        print(
            "❌ Ollama Cloud requires OLLAMA_API_KEY in .env",
            file=sys.stderr,
        )
        sys.exit(1)

    out1 = here / "output" / "decider_stage1.md"
    out2 = here / "output" / "decider_stage2.md"

    _run_stage(
        "Stage 1 — original priorities (budget ~$8k, ~120 guests, outdoor required)",
        PRIORITIES_STAGE1,
        base_url=base_url,
        model=model,
        api_key=api_key,
        out_path=out1,
    )
    _run_stage(
        "Stage 2 — shifted priorities (up to $15k, ~200 guests, elegant/grand)",
        PRIORITIES_STAGE2,
        base_url=base_url,
        model=model,
        api_key=api_key,
        out_path=out2,
    )

    print("─" * 60)
    print("💾 Artifacts")
    print(f"   📄 {out1}")
    print(f"   📄 {out2}")
    print("   📝 Edit REFLECTION_decider.md for Stage 3 submission text.")
    print("─" * 60 + "\n")


if __name__ == "__main__":
    main()
