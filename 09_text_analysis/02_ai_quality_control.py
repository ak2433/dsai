# 02_ai_quality_control.py
# AI-Assisted Text Quality Control
# Tim Fraser

# This script demonstrates how to use AI (Ollama or OpenAI) to perform quality control
# on AI-generated text reports. It implements quality control criteria including
# boolean accuracy checks and Likert scales for multiple quality dimensions.
# Students learn to design quality control prompts and structure AI outputs as JSON.

# pip install pandas requests python-dotenv

import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

AI_PROVIDER = "ollama"

PORT = 11434
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", f"http://127.0.0.1:{PORT}").rstrip("/")
OLLAMA_TAGS_URL = f"{OLLAMA_HOST}/api/tags"
OLLAMA_CHAT_URL = f"{OLLAMA_HOST}/api/chat"
OLLAMA_GENERATE_URL = f"{OLLAMA_HOST}/api/generate"
REQUEST_TIMEOUT = 300
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:latest")
# Detailed-rubric JSON needs more tokens on small local models
OLLAMA_NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", "2048"))

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"

_data_path = Path(__file__).resolve().parent / "data" / "sample_reports.txt"
with open(_data_path, "r", encoding="utf-8") as f:
    sample_text = f.read()

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

print("📝 Report for Quality Control:")
print("---")
print(report)
print("---\n")


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
        f"Ollama not reachable at {OLLAMA_HOST}. Start Ollama, pull a model, retry.\nLast error: {last_err}"
    )


def query_ollama_json_prompt(prompt: str, num_predict: int | None = None) -> str:
    if num_predict is None:
        num_predict = OLLAMA_NUM_PREDICT
    opts = {"temperature": 0.3, "num_predict": num_predict}
    errs: list[str] = []

    r_gen = requests.post(
        OLLAMA_GENERATE_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": opts,
        },
        timeout=REQUEST_TIMEOUT,
    )
    if r_gen.ok:
        out = (r_gen.json() or {}).get("response")
        if out:
            return out
        errs.append(f"generate: {str(r_gen.json())[:300]}")
    else:
        errs.append(f"generate HTTP {r_gen.status_code}: {r_gen.text[:400]}")

    r_chat = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "format": "json",
            "stream": False,
            "options": opts,
        },
        timeout=REQUEST_TIMEOUT,
    )
    if r_chat.ok:
        content = (r_chat.json().get("message") or {}).get("content")
        if content is not None:
            print("⚠️  Used /api/chat (generate had no usable text).\n")
            return content
        errs.append(f"chat: {str(r_chat.json())[:300]}")
    else:
        errs.append(f"chat HTTP {r_chat.status_code}: {r_chat.text[:400]}")

    raise RuntimeError("Ollama failed:\n" + "\n".join(errs))


def create_quality_control_prompt(report_text, source_data=None):
    """
    Detailed rubric only: anchored Likert anchors, numeric_consistency, issues[], evidence_notes.
    """
    instructions = (
        "You are a quality control validator for AI-generated reports. "
        "Evaluate the report and return ONLY valid JSON (no markdown fences)."
    )
    data_context = ""
    if source_data is not None:
        data_context = f"\n\nSource Data (ground truth for accuracy checks):\n{source_data}\n"

    criteria = """

DETAILED RUBRIC (when Source Data is provided, use it to judge accuracy and faithfulness):

**accurate** (boolean): false if ANY number, county, year, pollutant, or table label conflicts with Source Data or is invented.

**accuracy** (1-5): 1 = multiple contradictions with Source Data; 3 = mostly aligned but one ambiguous figure; 5 = all stated facts match Source Data.

**faithfulness** (1-5): 1 = claims not traceable to Source Data; 3 = mostly supported with minor overreach; 5 = quantitative claims map to Source Data.

**numeric_consistency** (1-5): 1 = several wrong percentages/counts vs. the table; 3 = one minor mismatch; 5 = cited figures match the table.

**relevance** (1-5): 1 = large unrelated sections; 3 = some filler; 5 = focused on the dataset.

**formality**, **clarity**, **succinctness**: 1 = poor, 3 = acceptable, 5 = strong.

**issues** (array of strings, max 6): each item "criterion: concrete observation".

**evidence_notes** (string): one or two sentences naming a table row used to judge accuracy, or state that no numbers were checked.

Return valid JSON:
{
  "accurate": true,
  "accuracy": 3,
  "formality": 3,
  "faithfulness": 3,
  "clarity": 3,
  "succinctness": 3,
  "relevance": 3,
  "numeric_consistency": 3,
  "issues": ["accuracy: example"],
  "evidence_notes": "short",
  "details": "0-80 word summary"
}
"""
    return f"{instructions}{data_context}\n\nReport Text to Validate:\n{report_text}{criteria}"


def query_ai_quality_control(prompt, provider=AI_PROVIDER):
    if provider == "ollama":
        ensure_ollama_available()
        return query_ollama_json_prompt(prompt, num_predict=OLLAMA_NUM_PREDICT)

    if provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in .env file.")
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a quality control validator. Always return valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.3,
            },
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    raise ValueError("Invalid provider. Use 'ollama' or 'openai'.")


def parse_quality_control_results(json_response):
    m = re.search(r"\{.*\}", json_response, re.DOTALL)
    if m:
        json_response = m.group(0)
    quality_data = json.loads(json_response)

    row = {
        "accurate": quality_data["accurate"],
        "accuracy": quality_data["accuracy"],
        "formality": quality_data["formality"],
        "faithfulness": quality_data["faithfulness"],
        "clarity": quality_data["clarity"],
        "succinctness": quality_data["succinctness"],
        "relevance": quality_data["relevance"],
        "details": quality_data.get("details", ""),
    }
    if "numeric_consistency" in quality_data:
        row["numeric_consistency"] = quality_data["numeric_consistency"]
    if "issues" in quality_data:
        iss = quality_data["issues"]
        row["issues"] = " | ".join(iss) if isinstance(iss, list) else str(iss)
    if "evidence_notes" in quality_data:
        row["evidence_notes"] = quality_data["evidence_notes"]

    df = pd.DataFrame([row])
    likert = [
        "accuracy",
        "formality",
        "faithfulness",
        "clarity",
        "succinctness",
        "relevance",
        "numeric_consistency",
    ]
    likert = [c for c in likert if c in df.columns]
    nums = df[[c for c in likert if c in df.columns]].apply(pd.to_numeric, errors="coerce")
    df["overall_score"] = round(float(nums.mean(axis=1).iloc[0]), 2)
    return df


quality_prompt = create_quality_control_prompt(report, source_data)

if AI_PROVIDER == "ollama":
    print(f"🤖 Querying Ollama ({OLLAMA_MODEL} at {OLLAMA_HOST}) — detailed rubric...\n")
else:
    print("🤖 Querying OpenAI...\n")

ai_response = query_ai_quality_control(quality_prompt, provider=AI_PROVIDER)
print("📥 AI Response (raw):")
print(ai_response)
print()

quality_results = parse_quality_control_results(ai_response)
overall_score = float(quality_results["overall_score"].iloc[0])

print("✅ Quality Control Results:")
print(quality_results)
print()
print(f"📊 Overall Quality Score (mean of Likert columns): {overall_score:.2f} / 5.0")
print(f"📊 Accuracy Check: {'PASS' if quality_results['accurate'].values[0] else 'FAIL'}\n")
print("💡 Compare with 01_manual_quality_control.py results.\n")
