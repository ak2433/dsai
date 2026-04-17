import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


AI_PROVIDER = "ollama"  # "ollama" | "openai"

PORT = 11434
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", f"http://127.0.0.1:{PORT}").rstrip("/")
OLLAMA_TAGS_URL = f"{OLLAMA_HOST}/api/tags"
OLLAMA_CHAT_URL = f"{OLLAMA_HOST}/api/chat"
OLLAMA_GENERATE_URL = f"{OLLAMA_HOST}/api/generate"
REQUEST_TIMEOUT = 300
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", "2048"))

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SAMPLE_REPORTS = _REPO_ROOT / "09_text_analysis" / "data" / "sample_reports.txt"


def ensure_ollama_available(max_wait_seconds: int = 20, poll_interval_seconds: float = 0.5) -> None:
    deadline = time.time() + max_wait_seconds
    last_err = None
    while time.time() < deadline:
        try:
            r = requests.get(OLLAMA_TAGS_URL, timeout=5)
            if r.ok:
                return
        except Exception as e:
            last_err = e
        time.sleep(poll_interval_seconds)
    raise RuntimeError(
        f"Ollama not reachable at {OLLAMA_HOST}. Start Ollama, run `ollama pull {OLLAMA_MODEL}`, retry.\n"
        f"Last error: {last_err}"
    )


def _ollama_keep_alive_body() -> dict:
    # Read after load_dotenv(): if set (e.g. "0"), Ollama unloads the model after this response.
    ka = os.environ.get("OLLAMA_KEEP_ALIVE")
    if ka is None:
        return {}
    if ka.isdigit():
        return {"keep_alive": int(ka)}
    return {"keep_alive": ka}


def query_ollama_json_prompt(prompt: str, num_predict: int = OLLAMA_NUM_PREDICT) -> str:
    opts = {"temperature": 0.3, "num_predict": num_predict}
    errs: list[str] = []

    gen_payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": opts,
        **_ollama_keep_alive_body(),
    }
    r_gen = requests.post(
        OLLAMA_GENERATE_URL,
        json=gen_payload,
        timeout=REQUEST_TIMEOUT,
    )
    if r_gen.ok:
        out = (r_gen.json() or {}).get("response")
        if out:
            return out
        errs.append(f"generate: no response field ({str(r_gen.json())[:300]})")
    else:
        errs.append(f"generate HTTP {r_gen.status_code}: {r_gen.text[:400]}")

    chat_payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "format": "json",
        "stream": False,
        "options": opts,
        **_ollama_keep_alive_body(),
    }
    r_chat = requests.post(
        OLLAMA_CHAT_URL,
        json=chat_payload,
        timeout=REQUEST_TIMEOUT,
    )
    if r_chat.ok:
        content = (r_chat.json().get("message") or {}).get("content")
        if content is not None:
            print("⚠️  Used /api/chat (generate had no usable text).\n")
            return content
        errs.append(f"chat: no content ({str(r_chat.json())[:300]})")
    else:
        errs.append(f"chat HTTP {r_chat.status_code}: {r_chat.text[:400]}")

    raise RuntimeError("Ollama failed:\n" + "\n".join(errs))


def create_quality_control_prompt(report_text: str, source_data: str | None = None) -> str:
    """Detailed rubric: anchored 1/3/5 rules, numeric_consistency, issues[], evidence_notes."""
    instructions = (
        "You are a quality control validator for AI-generated reports. "
        "Return ONLY valid JSON (no markdown fences, no text before or after the JSON object)."
    )

    data_block = ""
    if source_data and str(source_data).strip():
        data_block = (
            "\n\nSource Data (ground truth — use for accurate, accuracy, faithfulness, numeric_consistency):\n"
            f"{source_data}\n"
        )

    criteria = """

DETAILED RUBRIC (use these anchors when Source Data is provided):

**accurate** (boolean): false if ANY number, label, county, year, pollutant, or row label in the report conflicts with Source Data or is invented.

**accuracy** (1-5): 1 = multiple contradictions with Source Data; 3 = mostly aligned but one vague or ambiguous number; 5 = all stated facts match Source Data.

**faithfulness** (1-5): 1 = claims not traceable to Source Data (or invented); 3 = mostly supported but some overreach; 5 = every quantitative claim maps to Source Data.

**numeric_consistency** (1-5): 1 = several wrong percentages/counts vs. the table; 3 = one minor mismatch or rounding issue explained; 5 = all cited figures match the table.

**relevance** (1-5): 1 = large portions unrelated to the data; 3 = some off-topic filler; 5 = all paragraphs address the dataset.

**formality**, **clarity**, **succinctness**: use 1 = poor, 3 = acceptable, 5 = strong — and briefly justify in **issues** if any score is 1 or 2.

**issues** (array of strings, max 6): each item format "criterion: one concrete observation" (e.g. "accuracy: report says 40% for cars but table shows 36.1%").

**evidence_notes** (string, max 2 sentences): name one table row you relied on to judge accuracy, or say "no numeric claims to verify" if none.

Required JSON (no markdown fences):
{
  "accurate": true,
  "accuracy": 3,
  "formality": 3,
  "faithfulness": 3,
  "clarity": 3,
  "succinctness": 3,
  "relevance": 3,
  "numeric_consistency": 3,
  "issues": ["criterion: note"],
  "evidence_notes": "short",
  "details": "0-80 words"
}
"""
    return f"{instructions}{data_block}\n\nReport Text to Validate:\n{report_text}{criteria}"


def query_ai_quality_control(prompt: str, provider: str = AI_PROVIDER) -> str:
    if provider == "ollama":
        ensure_ollama_available()
        return query_ollama_json_prompt(prompt)

    if provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("Set OPENAI_API_KEY in .env (see ACTIVITY_openai_api_key.md).")
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": "You return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.3,
            },
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    raise ValueError("AI_PROVIDER must be 'ollama' or 'openai'.")


def parse_quality_control_results(json_response: str) -> pd.DataFrame:
    m = re.search(r"\{[\s\S]*\}", json_response)
    blob = m.group(0) if m else json_response.strip()
    d = json.loads(blob)

    row: dict = {}
    for k in [
        "accurate",
        "accuracy",
        "formality",
        "faithfulness",
        "clarity",
        "succinctness",
        "relevance",
        "details",
    ]:
        if k in d:
            row[k] = d[k]
    if "numeric_consistency" in d:
        row["numeric_consistency"] = d["numeric_consistency"]
    if "issues" in d:
        iss = d["issues"]
        row["issues"] = " | ".join(iss) if isinstance(iss, list) else str(iss)
    if "evidence_notes" in d:
        row["evidence_notes"] = d["evidence_notes"]

    df = pd.DataFrame([row])
    likert = [c for c in [
        "accuracy",
        "formality",
        "faithfulness",
        "clarity",
        "succinctness",
        "relevance",
        "numeric_consistency",
    ] if c in df.columns]
    if likert:
        nums = df[likert].apply(pd.to_numeric, errors="coerce")
        df["overall_score"] = round(float(nums.mean(axis=1).iloc[0]), 2)
    return df


def manual_qc_snapshot(report: str) -> pd.DataFrame:
    terms = ["emissions", "county", "year", "pollutant", "recommendations", "data"]
    present = sum(1 for t in terms if re.search(re.escape(t), report, re.I))
    return pd.DataFrame(
        [
            {
                "keywords_found_of_6": present,
                "has_digits": bool(re.search(r"\d", report)),
                "has_percent": bool(re.search(r"\d\s*%", report)),
            }
        ]
    )


def main() -> None:
    if not _SAMPLE_REPORTS.is_file():
        raise FileNotFoundError(
            f"Expected sample reports at {_SAMPLE_REPORTS}. Run from repo with 09_text_analysis/data present."
        )

    sample_text = _SAMPLE_REPORTS.read_text(encoding="utf-8")
    reports = [r.strip() for r in sample_text.split("\n\n") if r.strip()]
    report = reports[0]

    source_data = """White County, IL | 2015 | PM10 | Time Driven | hours
|type        |label_value |label_percent |
|:-----------|:-----------|:-------------|
|Light Truck |2.7 M       |51.8%         |
|Car/ Bike   |1.9 M       |36.1%         |
|Combo Truck |381.3 k     |7.3%          |
|Heavy Truck |220.7 k     |4.2%          |
|Bus         |30.6 k      |0.6%          |"""

    print("Rubric: detailed only (anchored Likert + numeric_consistency + issues + evidence_notes).\n")

    print("=== Sample report + manual snapshot ===\n---")
    print(report[:1200] + ("..." if len(report) > 1200 else ""))
    print("---\n")
    print(manual_qc_snapshot(report).to_string(index=False))
    print()

    print(f"=== AI QC ({AI_PROVIDER}, {OLLAMA_MODEL if AI_PROVIDER == 'ollama' else OPENAI_MODEL}) ===\n")

    prompt = create_quality_control_prompt(report, source_data)
    raw = query_ai_quality_control(prompt, provider=AI_PROVIDER)
    print("Raw JSON (truncated):", raw[:900] + ("..." if len(raw) > 900 else ""), "\n")

    qc = parse_quality_control_results(raw)
    print(qc.to_string())
    print()
    if "overall_score" in qc.columns:
        print(f"overall_score: {qc['overall_score'].iloc[0]} / 5.0 (mean of 7 Likert fields)")
    if "accurate" in qc.columns:
        print(f"accurate: {qc['accurate'].iloc[0]}")

    print(
        "\nManual checks count keywords; detailed AI QC ties scores to Source Data via issues/evidence_notes.\n"
        "If JSON is truncated, set env OLLAMA_NUM_PREDICT higher (default 2048).\n"
    )


if __name__ == "__main__":
    main()
