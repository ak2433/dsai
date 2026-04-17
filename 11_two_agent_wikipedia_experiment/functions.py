# functions.py
# Ollama agent + tool-calling helpers (standalone copy for two-agent Wikipedia experiment)
# Pairs with 08_function_calling/functions.R concepts
# Tim Fraser

# This script contains functions used for multi-agent orchestration with function calling in Python.

# 0. SETUP ###################################

## 0.1 Load Packages #################################

import json  # for working with JSON
import sys  # for stack frame inspection
import time  # for simple polling/retry

import requests  # for HTTP requests
import pandas as pd  # for data manipulation (df_as_text)

# pip install requests pandas

## 0.2 Configuration #################################

DEFAULT_MODEL = "smollm2:1.7b"
PORT = 11434
OLLAMA_HOST = f"http://localhost:{PORT}"
CHAT_URL = f"{OLLAMA_HOST}/api/chat"
REQUEST_TIMEOUT = 300
OLLAMA_TAGS_URL = f"{OLLAMA_HOST}/api/tags"


def ensure_ollama_available(max_wait_seconds: int = 15, poll_interval_seconds: float = 0.5) -> None:
    """Fail fast with a helpful message if Ollama isn't reachable."""
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
        "Ollama is not reachable at localhost:11434. "
        "Start Ollama (e.g. `ollama serve`), then retry.\n"
        f"Last error: {last_err}"
    )


# 1. AGENT FUNCTION ###################################


def agent(messages, model=DEFAULT_MODEL, output="text", tools=None, all=False, num_predict=500):
    """Run one chat completion, optionally with tools (single hop: model → tool execution)."""

    if tools is None:
        ensure_ollama_available()
        body = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": num_predict},
        }
        response = requests.post(CHAT_URL, json=body, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        result = response.json()
        return result["message"]["content"]

    ensure_ollama_available()
    body = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": False,
        "options": {"num_predict": num_predict},
    }
    response = requests.post(CHAT_URL, json=body, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    result = response.json()

    if "tool_calls" in result.get("message", {}):
        tool_calls = result["message"]["tool_calls"]
        for tool_call in tool_calls:
            func_name = tool_call["function"]["name"]
            raw_args = tool_call["function"].get("arguments", {})
            func_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            func = globals().get(func_name)
            if func is None:
                for depth in range(1, 6):
                    try:
                        frame = sys._getframe(depth)
                        func = frame.f_globals.get(func_name)
                        if func is not None:
                            break
                    except ValueError:
                        break
            if func:
                tool_output = func(**func_args)
                tool_call["output"] = tool_output

    if all:
        return result
    if "tool_calls" in result.get("message", {}):
        if output == "tools":
            return tool_calls
        return tool_calls[-1].get("output", result["message"]["content"])
    return result["message"]["content"]


def agent_run(role, task, tools=None, output="text", model=DEFAULT_MODEL, num_predict=500):
    """System prompt + one user task → `agent()` call."""
    messages = [
        {"role": "system", "content": role},
        {"role": "user", "content": task},
    ]
    return agent(messages=messages, model=model, output=output, tools=tools, num_predict=num_predict)


# 2. DATA CONVERSION FUNCTION ###################################


def df_as_text(df):
    """Convert a pandas DataFrame to a markdown table string."""
    return df.to_markdown(index=False)
