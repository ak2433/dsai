# two_agent_wikipedia_workflow.py
# Two-agent experiment: stock quote (tool) + Wikipedia context + report
# Tim Fraser

# Example: a rumor-style headline (“Jeff Bezos just sold Amazon”) → Agent 1 calls get_stock_quote(AMZN);
# Agent 1 decides what Wikipedia lookups help; Agent 2 summarizes what Amazon is (from Wikipedia)
# alongside the live quote. Self-contained: run from `11_two_agent_wikipedia_experiment/`.

# 0. Setup #################################

## 0.1 Load packages ############################

import json
import os
import re

import requests

## 0.2 Load helpers ############################

from functions import DEFAULT_MODEL, agent_run

MODEL = os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
NUM_PREDICT_TOOL = 220
NUM_PREDICT_GAP = 750
NUM_PREDICT_REPORT = 1400

# https://meta.wikimedia.org/wiki/User-Agent_policy
WIKI_HEADERS = {"User-Agent": "dsai-wikipedia-two-agent-demo/1.0 (educational script)"}

# Yahoo’s chart endpoint is unofficial; a browser-like User-Agent reduces blocked requests.
YAHOO_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


# 1. Custom tool: stock quote (Yahoo Finance chart API, no API key) ##################


def get_stock_quote(symbol: str) -> str:
    """
    Fetch a snapshot of the latest regular-market price for a ticker via Yahoo’s chart API.

    This is for education/demos only; Yahoo may rate-limit or change the API. Not investment advice.

    Parameters
    ----------
    symbol : str
        Exchange ticker, e.g. ``AMZN`` for Amazon.com Inc.

    Returns
    -------
    str
        Short human-readable summary, or an error message.
    """
    raw = (symbol or "").strip().upper()
    # Allow tickers like BRK.B
    sym = re.sub(r"[^A-Z0-9.]", "", raw)
    if not sym:
        return "Error: empty symbol. Use a ticker such as AMZN for Amazon."

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
    try:
        r = requests.get(
            url,
            params={"interval": "1d", "range": "1d"},
            headers=YAHOO_HEADERS,
            timeout=20,
        )
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        return f"Error fetching quote for {sym}: {e}"

    chart = payload.get("chart") or {}
    err = chart.get("error")
    if err:
        desc = err.get("description") if isinstance(err, dict) else err
        return f"Yahoo chart API error for {sym}: {desc}"

    results = chart.get("result") or []
    if not results:
        return f"No market data returned for symbol {sym}."

    meta = results[0].get("meta") or {}
    name = meta.get("longName") or meta.get("shortName") or sym
    price = meta.get("regularMarketPrice")
    prev = meta.get("previousClose")
    currency = meta.get("currency") or "?"
    exchange = meta.get("exchangeName") or meta.get("exchange") or "?"

    if price is None:
        return f"{name} ({sym}): no regularMarketPrice in response (market may be closed or symbol invalid)."

    lines = [
        f"Symbol: {sym}",
        f"Name (Yahoo): {name}",
        f"Exchange (if given): {exchange}",
        f"Regular-market price: {price} {currency}",
    ]
    if prev is not None:
        lines.append(f"Previous close (reference): {prev} {currency}")
    lines.append("(Source: Yahoo Finance chart API; snapshot for demo use only.)")
    return "\n".join(lines)


tool_get_stock_quote = {
    "type": "function",
    "function": {
        "name": "get_stock_quote",
        "description": (
            "Get a current regular-market stock price snapshot for a ticker symbol using Yahoo Finance "
            "(no API key). For Amazon the company, use symbol AMZN. Call exactly once with the right ticker."
        ),
        "parameters": {
            "type": "object",
            "required": ["symbol"],
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Ticker symbol, e.g. AMZN for Amazon.com Inc.",
                },
            },
        },
    },
}


# 2. Wikipedia helpers (feed Agent 2; not an Ollama tool in this demo) ##################


def _wiki_opensearch_title(phrase: str) -> str | None:
    url = "https://en.wikipedia.org/w/api.php"
    params = {"action": "opensearch", "search": phrase, "limit": 1, "namespace": 0, "format": "json"}
    r = requests.get(url, params=params, headers=WIKI_HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list) and len(data) >= 2 and isinstance(data[1], list) and len(data[1]) > 0:
        return str(data[1][0])
    return None


def wikipedia_plain_extract(title: str, sentences: int = 6) -> str:
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "exintro": True,
        "explaintext": True,
        "exsentences": sentences,
    }
    r = requests.get(url, params=params, headers=WIKI_HEADERS, timeout=20)
    r.raise_for_status()
    pages = (r.json().get("query") or {}).get("pages") or {}
    for _pid, page in pages.items():
        if page.get("missing"):
            return f"(No extract: missing article '{title}')"
        extract = (page.get("extract") or "").strip()
        if extract:
            return extract
    return f"(No extract for '{title}')"


def fetch_wikipedia_excerpts(queries: list[str], max_articles: int = 4, sentences: int = 6) -> str:
    blocks: list[str] = []
    for q in queries[:max_articles]:
        q = (q or "").strip()
        if not q:
            continue
        title = _wiki_opensearch_title(q) or q
        body = wikipedia_plain_extract(title, sentences=sentences)
        blocks.append(f"## Wikipedia — {title}\n{body}\n")
    if not blocks:
        return "(No Wikipedia excerpts retrieved.)\n"
    return "\n".join(blocks)


def parse_wikipedia_queries(agent_text: str) -> list[str]:
    text = (agent_text or "").strip()
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and "wikipedia_queries" in line:
            try:
                obj = json.loads(line)
                qs = obj.get("wikipedia_queries") or []
                return [str(x).strip() for x in qs if str(x).strip()]
            except json.JSONDecodeError:
                continue
    m = re.search(r"\{[^{}]*\"wikipedia_queries\"\s*:\s*\[[^\]]*\][^{}]*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            qs = obj.get("wikipedia_queries") or []
            return [str(x).strip() for x in qs if str(x).strip()]
        except json.JSONDecodeError:
            pass
    # Sensible default for the Amazon / headline demo
    return ["Amazon (company)", "Jeff Bezos"]


# 3. Agent roles ############################


ROLE_AGENT1_TOOL = """You are Agent 1 (market data). The user may describe a rumor, headline, or question
about a public company (e.g. “Jeff Bezos just sold Amazon”).

You MUST call get_stock_quote exactly once with the correct ticker. For Amazon.com the public stock, use
symbol AMZN. Do not answer the rumor or explain the company yet—only call the tool."""

ROLE_AGENT1_GAP = """You are still Agent 1. You see the output from get_stock_quote (price snapshot).

1) Briefly restate what the numbers show (symbol, price, currency).
2) Comment that a sensational headline may be unverified; the quote is only a market snapshot, not proof
   of any corporate event.
3) Say what Wikipedia could clarify for a reader—for example what the company is, who founded it, or how
   it relates to the headline.
4) On the LAST line only, output valid JSON (no markdown fences):
   {"wikipedia_queries": ["phrase1", "phrase2", "phrase3"]}
   Include “Amazon (company)” (or the right company article) when the story is about Amazon. 2–4 phrases."""

ROLE_AGENT2_REPORT = """You are Agent 2 (news desk / explainer). You receive:
- The user’s headline or question
- Agent 1’s notes and stock snapshot context
- Wikipedia lead excerpts (already fetched)

Write a short, readable piece (4–8 paragraphs):
- Start with a plain-language summary of **what the company is**, using Wikipedia (name the article titles
  you rely on, in parentheses).
- Incorporate the **stock snapshot** from Agent 1 as context only—it does not confirm rumors.
- If the user’s claim sounds like breaking news you cannot verify from Wikipedia or the quote alone, say so clearly.

Do not invent facts; stick to the excerpts plus the quoted stock lines. If an excerpt is thin, say so briefly."""


# 4. Workflow ############################


def run_two_agent_workflow(user_task: str) -> tuple[str, str, str]:
    """
    Agent 1: tool call (stock quote) + gap analysis + Wikipedia search phrases.
    Agent 2: report (company summary from Wikipedia + quote context).

    Returns
    -------
    agent1_display, gap_analysis, agent2_report
    """
    tool_result = agent_run(
        role=ROLE_AGENT1_TOOL,
        task=user_task,
        tools=[tool_get_stock_quote],
        output="tools",
        model=MODEL,
        num_predict=NUM_PREDICT_TOOL,
    )
    quote_text = ""
    if isinstance(tool_result, list) and len(tool_result) > 0:
        quote_text = str(tool_result[0].get("output", ""))
    else:
        quote_text = "(No tool output—check that your Ollama model supports function calling.)"

    gap_task = f"User message:\n{user_task}\n\nStock tool output:\n{quote_text}\n"
    gap_analysis = agent_run(
        role=ROLE_AGENT1_GAP,
        task=gap_task,
        tools=None,
        model=MODEL,
        num_predict=NUM_PREDICT_GAP,
    )

    agent1_display = (
        "=== Agent 1 — tool output (get_stock_quote) ===\n"
        f"{quote_text}\n\n"
        "=== Agent 1 — analysis + Wikipedia targets ===\n"
        f"{gap_analysis}\n"
    )

    queries = parse_wikipedia_queries(gap_analysis)
    wiki_blob = fetch_wikipedia_excerpts(queries, max_articles=4, sentences=8)

    report_task = (
        f"## User message\n{user_task}\n\n"
        f"## Agent 1\n{gap_analysis}\n\n"
        f"## Wikipedia excerpts\n{wiki_blob}\n"
    )
    report = agent_run(
        role=ROLE_AGENT2_REPORT,
        task=report_task,
        tools=None,
        model=MODEL,
        num_predict=NUM_PREDICT_REPORT,
    )

    return agent1_display, gap_analysis, report


if __name__ == "__main__":
    demo = (
        'Headline going around: "Jeff Bezos just sold Amazon." '
        "Fetch the relevant stock snapshot and explain what Amazon is with background context."
    )
    block1, _g, rep2 = run_two_agent_workflow(demo)
    print(block1)
    print("=== Agent 2 — report (Wikipedia + stock snapshot) ===")
    print(rep2)
