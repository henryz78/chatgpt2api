from __future__ import annotations

import json
import re
import uuid
from typing import Any, Iterator


# ── System prompt injected when tools are present ──────────────────────────────

_TOOL_SYSTEM_PROMPT = """\
## Tool Use Instructions — READ CAREFULLY

You are operating in a controlled environment where you have NO built-in tools, \
NO web search, NO code execution, and NO internet access of your own. \
You MUST NOT use any built-in capabilities to answer questions.

The ONLY tools you may use are the external functions listed below, \
which are provided by the application.

### Rules (non-negotiable)

1. If answering a request requires calling a tool, output **ONLY** the JSON block \
   below — nothing before it, nothing after it, no markdown prose.
2. Do NOT use your own web search, browsing, or any other built-in feature.
3. Do NOT fabricate tool results — wait for the tool response in the next message.
4. After receiving tool results (marked `[Tool result ...]`), reply normally.

### Output format for tool calls

```json
{{"tool_calls": [{{"name": "<function_name>", "arguments": <arguments_object>}}]}}
```

Parallel calls: include multiple objects in the array.

### Concrete example

User: Look up the price of AAPL.
Assistant (CORRECT — output only the JSON, nothing else):
```json
{{"tool_calls": [{{"name": "get_stock_price", "arguments": {{"symbol": "AAPL"}}}}]}}
```

### Available tools

{tools_json}
"""

# ── Parsers ────────────────────────────────────────────────────────────────────

# Primary: ```json ... ``` block containing {"tool_calls": [...]}
_CODE_BLOCK_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\"tool_calls\".*?\})\s*```",
    re.DOTALL | re.IGNORECASE,
)

# Secondary: bare {"tool_calls": [...]} anywhere in the text
_BARE_TOOL_CALLS_RE = re.compile(
    r'\{\s*"tool_calls"\s*:\s*(\[.*?\])\s*\}',
    re.DOTALL,
)

# Tertiary: legacy <tool_calls>[...]</tool_calls> format
_XML_TOOL_CALLS_RE = re.compile(
    r"<tool_calls>\s*(.*?)\s*</tool_calls>",
    re.DOTALL,
)


def _build_tool_calls(raw: list[Any]) -> list[dict[str, Any]] | None:
    result: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict) or "name" not in item:
            continue
        arguments = item.get("arguments", {})
        result.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": str(item["name"]),
                "arguments": (
                    arguments
                    if isinstance(arguments, str)
                    else json.dumps(arguments, ensure_ascii=False)
                ),
            },
        })
    return result or None


def parse_tool_calls(text: str) -> list[dict[str, Any]] | None:
    """Try multiple patterns to extract tool calls from model output."""
    # 1. Code block with {"tool_calls": [...]}
    m = _CODE_BLOCK_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(1))
            calls = obj.get("tool_calls") if isinstance(obj, dict) else None
            if isinstance(calls, list):
                result = _build_tool_calls(calls)
                if result:
                    return result
        except (json.JSONDecodeError, ValueError):
            pass

    # 2. Bare {"tool_calls": [...]}
    m = _BARE_TOOL_CALLS_RE.search(text)
    if m:
        try:
            calls = json.loads(m.group(1))
            if isinstance(calls, list):
                result = _build_tool_calls(calls)
                if result:
                    return result
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. Legacy XML-style <tool_calls>[...]</tool_calls>
    m = _XML_TOOL_CALLS_RE.search(text)
    if m:
        try:
            calls = json.loads(m.group(1))
            if isinstance(calls, list):
                result = _build_tool_calls(calls)
                if result:
                    return result
        except (json.JSONDecodeError, ValueError):
            pass

    return None


# ── Public helpers ─────────────────────────────────────────────────────────────

def tools_system_message(tools: list[dict[str, Any]]) -> dict[str, Any]:
    prompt = _TOOL_SYSTEM_PROMPT.format(
        tools_json=json.dumps(tools, ensure_ascii=False, indent=2)
    )
    return {"role": "system", "content": prompt}


def tool_calls_response(
    model: str,
    tool_calls: list[dict[str, Any]],
    created: int | None = None,
) -> dict[str, Any]:
    import time
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": created or int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls,
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def stream_tool_calls_chunks(
    model: str,
    tool_calls: list[dict[str, Any]],
    completion_id: str,
    created: int,
) -> Iterator[dict[str, Any]]:
    def _chunk(delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
        return {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    yield _chunk({"role": "assistant", "content": None})

    for idx, tc in enumerate(tool_calls):
        fn = tc["function"]
        yield _chunk({
            "tool_calls": [{
                "index": idx,
                "id": tc["id"],
                "type": "function",
                "function": {"name": fn["name"], "arguments": ""},
            }]
        })
        args = fn.get("arguments") or "{}"
        chunk_size = 16
        for start in range(0, len(args), chunk_size):
            yield _chunk({
                "tool_calls": [{
                    "index": idx,
                    "function": {"arguments": args[start: start + chunk_size]},
                }]
            })

    yield _chunk({}, "tool_calls")


def has_tools(body: dict[str, Any]) -> bool:
    tools = body.get("tools")
    return isinstance(tools, list) and len(tools) > 0


def normalize_tool_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert tool_calls / tool-result messages to plain text ChatGPT can understand."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        role = str(msg.get("role") or "user")

        if role == "assistant" and msg.get("tool_calls"):
            parts: list[str] = []
            for tc in msg["tool_calls"]:
                fn = tc.get("function") or {}
                name = str(fn.get("name") or "unknown")
                raw_args = fn.get("arguments") or "{}"
                try:
                    args_obj = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    args_text = json.dumps(args_obj, ensure_ascii=False)
                except Exception:
                    args_text = str(raw_args)
                parts.append(f"[Called tool `{name}` with arguments: {args_text}]")
            existing = str(msg.get("content") or "").strip()
            content = "\n".join(parts)
            if existing:
                content = f"{existing}\n{content}"
            result.append({"role": "assistant", "content": content})

        elif role == "tool":
            tool_call_id = str(msg.get("tool_call_id") or "")
            content = str(msg.get("content") or "")
            prefix = f"[Tool result (id={tool_call_id}): " if tool_call_id else "[Tool result: "
            result.append({"role": "user", "content": f"{prefix}{content}]"})

        else:
            result.append(msg)

    return result
