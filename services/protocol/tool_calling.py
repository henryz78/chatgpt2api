"""
Tool calling adapter for chatgpt2api.

Design reference: ds-free-api (Rust, for DeepSeek)
  - Separate format-block / defs-text / instruction-text, mirroring ds-free-api's ToolContext
  - Dynamic examples built from actual tool names in the request
  - tool_choice variants handled explicitly (auto / required / named)
  - parallel_tool_calls: false support
  - Consistent format between system-prompt injection and history normalisation
  - JSON repair (invalid backslashes, bare unquoted keys)

Key difference vs ds-free-api: DeepSeek has native trained tokens; GPT-4 via the
web API does not.  We therefore inject everything as plain text in the system
message and rely on the model to follow instructions.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any, Iterator


# ── Tag constants (match ds-free-api's canonical XML style) ───────────────────
TOOL_CALL_START = "<tool_calls>"
TOOL_CALL_END = "</tool_calls>"
TOOL_RESULT_START = "<tool_results>"
TOOL_RESULT_END = "</tool_results>"


# ── Known example arguments (mirrors ds-free-api's example_args) ──────────────
_EXAMPLE_ARGS: dict[str, dict[str, Any]] = {
    "get_weather": {"city": "Beijing"},
    "get_time": {"timezone": "Asia/Shanghai"},
    "get_stock_price": {"symbol": "AAPL"},
    "search": {"query": "latest AI news"},
    "read_file": {"file_path": "/path/to/file"},
    "write_file": {"file_path": "/path/to/file", "content": "hello"},
    "execute_command": {"command": "ls -la"},
    "list_files": {"path": "."},
    "get_order_status": {"order_id": "ORD-12345"},
    "send_email": {"to": "user@example.com", "subject": "Hi", "body": "Hello"},
}

_NESTED_EXAMPLE_ARGS: dict[str, Any] = {
    "config": {"enabled": True, "items": ["a", "b"]}
}


def _example_args_for(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Build a realistic example argument dict for a given tool."""
    if name in _EXAMPLE_ARGS:
        return _EXAMPLE_ARGS[name]
    # Derive from schema properties
    props = schema.get("parameters", {}).get("properties", {})
    example: dict[str, Any] = {}
    for k, v in props.items():
        t = v.get("type", "string")
        if t == "string":
            example[k] = f"<{k}>"
        elif t == "integer":
            example[k] = 0
        elif t == "number":
            example[k] = 0.0
        elif t == "boolean":
            example[k] = True
        elif t == "array":
            example[k] = []
        else:
            example[k] = {}
    return example or {"key": "value"}


def _tool_call_line(name: str, args: dict[str, Any]) -> str:
    return (
        f'{TOOL_CALL_START}'
        f'[{{"name": "{name}", "arguments": {json.dumps(args, ensure_ascii=False)}}}]'
        f'{TOOL_CALL_END}'
    )


def _build_format_block(tools: list[dict[str, Any]], parallel: bool = True) -> str:
    """Build the format-specification block (rules + examples)."""
    lines: list[str] = []

    # ── Rules ──────────────────────────────────────────────────────────────
    lines += [
        "**工具调用格式 — 请严格遵守：**",
        "",
        "将 JSON 数组包裹在工具调用标记中：",
        "",
        f"{TOOL_CALL_START}[{{\"name\": \"工具名\", \"arguments\": {{参数JSON}}}}]{TOOL_CALL_END}",
        "",
        "**规则：**",
        "",
        "**核心：决定调用工具时，你的响应中只允许出现工具调用文本本身，禁止任何解释、前缀、总结等额外内容。**",
        "",
        f"1. JSON 数组必须以 `{TOOL_CALL_START}` 开头、以 `{TOOL_CALL_END}` 结尾，完整包裹在标记内。",
        "2. 所有工具调用必须放在**一个** JSON 数组中，多个调用用逗号分隔。",
        f"3. 输出 `{TOOL_CALL_END}` 后**立即停止**，不得添加后续文本或说明。",
        "4. 不要将工具调用包裹在 markdown 代码块中。",
        "5. 字符串参数值必须用**双引号**包裹（JSON 标准）。",
        f"6. 决定调用工具时，输出的**第一个非空白字符**必须是 `{TOOL_CALL_START}`。",
        "7. 不要使用你内置的 web search、browsing 或任何其他自带功能 — 仅使用下方列出的工具。",
        "8. 工具结果会在下一条用户消息中以 `<tool_results>` 标签返回，收到后正常继续回答。",
    ]

    if not parallel:
        lines.append("9. **一次只能调用一个工具。**")

    lines.append("")

    # ── Examples (use actual tool names) ───────────────────────────────────
    funcs = [t["function"] for t in tools if t.get("type") == "function" and "function" in t]

    if funcs:
        lines.append("**正确示例：**")
        lines.append("")

        a = funcs[0]
        a_args = _example_args_for(a["name"], a)
        lines.append(f"**示例A** — 调用单个工具：")
        lines.append(_tool_call_line(a["name"], a_args))
        lines.append("")

        if len(funcs) >= 2:
            b = funcs[1]
            b_args = _example_args_for(b["name"], b)
            items = ", ".join([
                f'{{"name": "{a["name"]}", "arguments": {json.dumps(a_args, ensure_ascii=False)}}}',
                f'{{"name": "{b["name"]}", "arguments": {json.dumps(b_args, ensure_ascii=False)}}}',
            ])
            lines.append("**示例B** — 同时调用多个工具（一个数组包含全部调用）：")
            lines.append(f"{TOOL_CALL_START}[{items}]{TOOL_CALL_END}")
            lines.append("")

    return "\n".join(lines)


def _build_defs_text(tools: list[dict[str, Any]]) -> str:
    """Format tool definitions for injection into the system prompt."""
    lines = ["你可以使用以下工具："]
    for t in tools:
        if t.get("type") != "function":
            continue
        fn = t.get("function", {})
        name = fn.get("name", "")
        desc = (fn.get("description") or "").strip()
        params = json.dumps(fn.get("parameters", {}), ensure_ascii=False)
        example_args = _example_args_for(name, fn)
        call_ex = _tool_call_line(name, example_args)
        desc_block = f"~~~markdown\n  {desc}\n~~~\n" if desc else "  无描述\n"
        lines.append(
            f"- **{name}** (function):\n"
            f"  - 调用方法: `{call_ex}`\n"
            f"  - 参数 schema: {params}\n"
            f"  - 说明:\n{desc_block}"
        )
    return "\n".join(lines)


def _build_instruction_text(
    tool_choice: Any,
    tools: list[dict[str, Any]],
) -> str | None:
    """Map tool_choice to a human-readable instruction line (mirrors ds-free-api)."""
    if tool_choice is None or tool_choice == "auto":
        return None

    if isinstance(tool_choice, str):
        if tool_choice == "required":
            return "**注意：你必须调用一个或多个工具，不能直接回答。**"
        if tool_choice == "none":
            return None  # caller should skip tool injection entirely
        return None

    if isinstance(tool_choice, dict):
        # {"type": "function", "function": {"name": "..."}}
        t = tool_choice.get("type")
        if t == "function":
            fn_name = (tool_choice.get("function") or {}).get("name", "")
            if fn_name:
                return f"**注意：你必须调用 '{fn_name}' 工具。**"
    return None


# ── Public: build the system message to inject ────────────────────────────────

def tools_system_message(
    tools: list[dict[str, Any]],
    tool_choice: Any = None,
    parallel_tool_calls: bool = True,
) -> dict[str, Any]:
    """
    Build a system message that teaches the model the tool-calling protocol.
    Mirrors ds-free-api's ToolContext (format_block + defs_text + instruction_text).
    """
    # tool_choice == "none" → caller should not call this function
    if tool_choice == "none":
        return {"role": "system", "content": ""}

    format_block = _build_format_block(tools, parallel=parallel_tool_calls)
    defs_text = _build_defs_text(tools)
    instruction_text = _build_instruction_text(tool_choice, tools)

    sections: list[str] = [
        "## 工具调用\n",
        f"### 格式规范\n{format_block}",
        f"### 工具定义\n{defs_text}",
    ]
    if instruction_text:
        sections.append(f"### 调用指令\n{instruction_text}")

    prompt = "\n\n".join(sections)
    return {"role": "system", "content": prompt}


# ── JSON repair (ported from ds-free-api's repair_json) ───────────────────────

def _repair_invalid_backslashes(s: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            nxt = s[i + 1]
            if nxt in ('"', "\\", "/", "b", "f", "n", "r", "t", "u"):
                out.append(c)
                out.append(nxt)
                i += 2
            else:
                out.append("\\\\")
                out.append(nxt)
                i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _repair_unquoted_keys(s: str) -> str:
    """Quote bare identifier keys in JSON objects."""
    return re.sub(
        r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)',
        lambda m: f'{m.group(1)}"{m.group(2)}"{m.group(3)}',
        s,
    )


def _repair_json(s: str) -> str | None:
    step1 = _repair_invalid_backslashes(s)
    try:
        json.loads(step1)
        return step1
    except (json.JSONDecodeError, ValueError):
        pass
    step2 = _repair_unquoted_keys(step1)
    try:
        json.loads(step2)
        return step2
    except (json.JSONDecodeError, ValueError):
        pass
    return None


# ── Parser ────────────────────────────────────────────────────────────────────

# Primary: <tool_calls>[...]</tool_calls>
_XML_RE = re.compile(
    r"<tool_calls>\s*(.*?)\s*</tool_calls>",
    re.DOTALL,
)

# Secondary: {"tool_calls": [...]} (possibly inside ```json ... ```)
_BARE_TOOL_CALLS_RE = re.compile(
    r'```(?:json)?\s*(\{[^`]*?"tool_calls"[^`]*?\})\s*```|'
    r'(\{"tool_calls"\s*:\s*\[.*?\](?:\s*\})?)',
    re.DOTALL,
)

# Tertiary: single-object {"name": ..., "arguments": ...}
_SINGLE_CALL_RE = re.compile(
    r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}',
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


def _parse_json_array(raw: str) -> list[Any] | None:
    """Parse a JSON array with repair fallback."""
    raw = raw.strip()
    # ensure it starts with [
    arr_start = raw.find("[")
    if arr_start == -1:
        return None
    arr_end = raw.rfind("]")
    if arr_end == -1:
        return None
    json_str = raw[arr_start: arr_end + 1]
    if json_str.strip() == "[]":
        return None
    try:
        result = json.loads(json_str)
        if isinstance(result, list):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    repaired = _repair_json(json_str)
    if repaired:
        try:
            result = json.loads(repaired)
            if isinstance(result, list):
                return result
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def parse_tool_calls(text: str) -> list[dict[str, Any]] | None:
    """
    Try multiple patterns to extract tool calls from model output.
    Pattern priority (mirrors ds-free-api):
      1. <tool_calls>[...]</tool_calls>
      2. {"tool_calls": [...]} (bare or in ```json block)
      3. Single {\"name\":...,\"arguments\":...} object
    """
    # 1. Canonical XML format
    m = _XML_RE.search(text)
    if m:
        arr = _parse_json_array(m.group(1))
        if arr is not None:
            result = _build_tool_calls(arr)
            if result:
                return result

    # 2. {"tool_calls": [...]} wrapper (bare or fenced)
    m = _BARE_TOOL_CALLS_RE.search(text)
    if m:
        raw_obj = m.group(1) or m.group(2)
        if raw_obj:
            try:
                obj = json.loads(raw_obj)
            except (json.JSONDecodeError, ValueError):
                repaired = _repair_json(raw_obj)
                obj = json.loads(repaired) if repaired else None
            if isinstance(obj, dict):
                calls = obj.get("tool_calls")
                if isinstance(calls, list):
                    result = _build_tool_calls(calls)
                    if result:
                        return result

    # 3. Bare single-call object
    m = _SINGLE_CALL_RE.search(text)
    if m:
        name = m.group(1)
        args_str = m.group(2)
        try:
            args = json.loads(args_str)
        except (json.JSONDecodeError, ValueError):
            repaired = _repair_json(args_str)
            args = json.loads(repaired) if repaired else {}
        result = _build_tool_calls([{"name": name, "arguments": args}])
        if result:
            return result

    return None


# ── Public helpers ─────────────────────────────────────────────────────────────

def has_tools(body: dict[str, Any]) -> bool:
    tools = body.get("tools")
    tc = body.get("tool_choice", "auto")
    if not isinstance(tools, list) or not tools:
        return False
    if tc == "none":
        return False
    return True


def tool_calls_response(
    model: str,
    tool_calls: list[dict[str, Any]],
    created: int | None = None,
) -> dict[str, Any]:
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
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
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


def normalize_tool_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert tool_calls / tool-result messages to a format GPT-4 can understand.

    Mirrors ds-free-api's format_assistant / format_tool:
    - assistant.tool_calls  → same <tool_calls>[...]</tool_calls> notation
      (model sees it already "outputted" that format, so it stays consistent)
    - role=tool             → <tool_results>[...]</tool_results> in a user message
    """
    result: list[dict[str, Any]] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = str(msg.get("role") or "user")

        if role == "assistant" and msg.get("tool_calls"):
            # Reconstruct the same XML the model would have produced
            items: list[str] = []
            for tc in msg["tool_calls"]:
                fn = tc.get("function") or {}
                name = str(fn.get("name") or "unknown")
                raw_args = fn.get("arguments") or "{}"
                try:
                    args_obj = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    args_text = json.dumps(args_obj, ensure_ascii=False)
                except Exception:
                    args_text = str(raw_args)
                items.append(f'{{"name": "{name}", "arguments": {args_text}}}')
            tool_call_text = f"{TOOL_CALL_START}[{', '.join(items)}]{TOOL_CALL_END}"
            existing = str(msg.get("content") or "").strip()
            content = f"{existing}\n{tool_call_text}" if existing else tool_call_text
            result.append({"role": "assistant", "content": content})
            i += 1

        elif role == "tool":
            # Group consecutive tool messages (mirrors ds-free-api's tool batching)
            tool_outputs: list[str] = []
            while i < len(messages) and str(messages[i].get("role") or "") == "tool":
                tm = messages[i]
                tool_call_id = str(tm.get("tool_call_id") or "")
                content = str(tm.get("content") or "")
                tool_outputs.append(
                    f'{{"tool_call_id": "{tool_call_id}", "output": {json.dumps(content, ensure_ascii=False)}}}'
                )
                i += 1
            inner = ", ".join(tool_outputs)
            result.append({
                "role": "user",
                "content": f"{TOOL_RESULT_START}[{inner}]{TOOL_RESULT_END}",
            })

        else:
            result.append(msg)
            i += 1

    return result
