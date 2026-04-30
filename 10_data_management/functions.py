# functions.py
# Shared helpers for fixer_csv.py (Ollama /api/chat + dataframe chunks)
# Tim Fraser

from __future__ import annotations

import json
from typing import Any

import httpx
import pandas as pd

REQUEST_TIMEOUT = 600.0


def split_df_into_row_chunks(df: pd.DataFrame, rows_per_batch: int) -> list[pd.DataFrame]:
    """Return contiguous row slices of df, each with at most rows_per_batch rows."""
    n = max(1, int(rows_per_batch))
    out: list[pd.DataFrame] = []
    for i in range(0, len(df), n):
        out.append(df.iloc[i : i + n].copy())
    return out


def parse_function_arguments(raw: Any) -> dict[str, Any]:
    """Normalize Ollama tool `function.arguments` (JSON string or dict) to a dict."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return {}
        return json.loads(s)
    return {}


def ollama_chat_once(
    ollama_host: str,
    ollama_key: str,
    ollama_model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    format: str | None = None,
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
    """
    Single non-streaming POST to Ollama /api/chat (local or cloud).

    Returns the parsed JSON body (includes `message` with optional `tool_calls`).
    """
    base = (ollama_host or "").strip().rstrip("/")
    url = f"{base}/api/chat"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if (ollama_key or "").strip():
        headers["Authorization"] = f"Bearer {ollama_key.strip()}"

    body: dict[str, Any] = {
        "model": ollama_model,
        "messages": messages,
        "stream": False,
    }
    if tools:
        body["tools"] = tools
    if format is not None:
        body["format"] = format
    opts: dict[str, Any] = {}
    if max_output_tokens is not None:
        opts["num_predict"] = int(max_output_tokens)
    if opts:
        body["options"] = opts

    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        r = client.post(url, headers=headers, json=body)
        r.raise_for_status()
        return r.json()
