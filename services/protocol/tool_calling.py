"""
Tool calling adapter for chatgpt2api.

Design reference: ds-free-api (Rust, for DeepSeek)
  - Separate format-block / defs-text / instruction-text, mirroring ds-free-api's ToolContext
  - Dynamic examples built from actual tool names in the request
  - tool_choice variants handled explicitly (auto / required / named)
  - parallel_tool_calls: false support
  - Consistent format between system-prompt injection and history normalisation
  - JSON repair (invalid backslashes, bare unquoted keys, missing brackets)
  - Multi-tag parser: <tool_calls>, ds-free-api DeepSeek tokens, <invoke>, bare JSON

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
        elif t == "object":
            example[k] = {"key": "value"}
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
        "2. 所有工具调用必须放在**一个** JSON 数组中，多个调用用逗号分隔。整个响应中只允许出现**一个**工具调用块。",
        f"3. 输出 `{TOOL_CALL_END}` 后**立即停止**，不得添加后续文本或说明。",
        "4. 不要将工具调用包裹在 markdown 代码块（` ``` `）中。",
        "5. 字符串参数值必须用**双引号**包裹（JSON 标准）。",
        f"6. 决定调用工具时，输出的**第一个非空白字符**必须是 `{TOOL_CALL_START}`。",
        "7. 不要使用你内置的 web search、browsing 或任何其他自带功能 — 仅使用下方列出的工具。",
        "8. 工具结果会在下一条用户消息中以 `<tool_results>` 标签返回，收到后正常继续回答。",
        "9. **不要把工具调用放进思考内容、推理过程或任何解释性文本中。** 工具调用只能作为最终输出的唯一内容。",
        "10. 嵌套对象/数组参数按 JSON 标准展开，例如：`{\"config\": {\"enabled\": true, \"items\": [\"a\", \"b\"]}}`。",
    ]

    if not parallel:
        lines.append("11. **一次只能调用一个工具。**")

    lines.append("")

    funcs = [t["function"] for t in tools if t.get("type") == "function" and "function" in t]

    if funcs:
        lines.append("**正确示例：**")
        lines.append("")

        a = funcs[0]
        a_args = _example_args_for(a["name"], a)
        lines.append("**示例A** — 调用单个工具：")
        lines.append(_tool_call_line(a["name"], a_args))
        lines.append("")

        if len(funcs) >= 2 and parallel:
            b = funcs[1]
            b_args = _example_args_for(b["name"], b)
            items_ab = ", ".join([
                f'{{"name": "{a["name"]}", "arguments": {json.dumps(a_args, ensure_ascii=False)}}}',
                f'{{"name": "{b["name"]}", "arguments": {json.dumps(b_args, ensure_ascii=False)}}}',
            ])
            lines.append("**示例B** — 同时调用两个工具（一个数组包含全部调用）：")
            lines.append(f"{TOOL_CALL_START}[{items_ab}]{TOOL_CALL_END}")
            lines.append("")

        if len(funcs) >= 3 and parallel:
            c = funcs[2]
            c_args = _example_args_for(c["name"], c)
            a_args2 = _example_args_for(a["name"], a)
            b_args2 = _example_args_for(b["name"], b)  # type: ignore[possibly-undefined]
            items_abc = ", ".join([
                f'{{"name": "{a["name"]}", "arguments": {json.dumps(a_args2, ensure_ascii=False)}}}',
                f'{{"name": "{b["name"]}", "arguments": {json.dumps(b_args2, ensure_ascii=False)}}}',  # type: ignore[possibly-undefined]
                f'{{"name": "{c["name"]}", "arguments": {json.dumps(c_args, ensure_ascii=False)}}}',
            ])
            lines.append("**示例C** — 同时调用三个工具（并行调用）：")
            lines.append(f"{TOOL_CALL_START}[{items_abc}]{TOOL_CALL_END}")
            lines.append("")

        # Nested params example
        lines.append("**示例D** — 含嵌套参数：")
        lines.append(
            f'{TOOL_CALL_START}[{{"name": "{a["name"]}", "arguments": '
            f'{json.dumps(_NESTED_EXAMPLE_ARGS, ensure_ascii=False)}}}]{TOOL_CALL_END}'
        )
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
            return None
        return None

    if isinstance(tool_choice, dict):
        t = tool_choice.get("type")
        if t == "function":
            fn_name = (tool_choice.get("function") or {}).get("name", "")
            if fn_name:
                return f"**注意：你必须调用 '{fn_name}' 工具，不能调用其他工具，也不能直接回答。**"
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


def _repair_missing_array_bracket(s: str) -> str:
    """If the string is a bare object like {...}, wrap it in [...]."""
    s2 = s.strip()
    if s2.startswith("{") and not s2.startswith("["):
        return f"[{s2}]"
    return s


def _repair_json(s: str) -> str | None:
    """Try progressively more aggressive repairs and return the first valid JSON string."""
    candidates = [s]

    step1 = _repair_invalid_backslashes(s)
    candidates.append(step1)

    step2 = _repair_unquoted_keys(step1)
    candidates.append(step2)

    step3 = _repair_unquoted_keys(s)
    candidates.append(step3)

    for candidate in candidates:
        try:
            json.loads(candidate)
            return candidate
        except (json.JSONDecodeError, ValueError):
            pass

    return None


# ── Code-fence stripping ───────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    """Remove triple-backtick code blocks.

    WARNING: only call this as a *fallback* after parsing the original text.
    Tool arguments may legitimately contain triple backticks (e.g. a write_file
    tool carrying markdown content).  Stripping unconditionally turns those
    valid calls into parse failures.
    """
    return _FENCE_RE.sub("", text)


# ── Parser patterns ────────────────────────────────────────────────────────────

# 1. Canonical: <tool_calls>[...]</tool_calls>
_XML_RE = re.compile(
    r"<tool_calls>\s*(.*?)\s*</tool_calls>",
    re.DOTALL,
)

# 2a. DeepSeek style (primary): <|tool▁calls▁begin|>...<|tool▁calls▁end|>
#     with fuzzy matching for ▁ vs _ and ｜ vs |
_DS_BEGIN_RE = re.compile(
    r"<[|｜]tool[▁_]calls[▁_]begin[|｜]>(.*?)<[|｜]tool[▁_]calls[▁_]end[|｜]>",
    re.DOTALL,
)

# 2b. DeepSeek individual call segment: <|tool_call_begin|>...<|tool_call_end|>
_DS_CALL_RE = re.compile(
    r"<[|｜]tool[▁_]call[▁_]begin[|｜](.*?)<[|｜]tool[▁_]call[▁_]end[|｜]>",
    re.DOTALL,
)

# 3. Simple <tool_call>...</tool_call>
_SIMPLE_TAG_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL,
)

# 4. {"tool_calls": [...]} bare (NOT inside code fences — fences stripped before use)
_BARE_TOOL_CALLS_RE = re.compile(
    r'(\{"tool_calls"\s*:\s*\[.*?\](?:\s*\})?)',
    re.DOTALL,
)

# 5. Single-object {"name": ..., "arguments": ...}
_SINGLE_CALL_RE = re.compile(
    r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{.*?\}|\[.*?\]|".*?")\s*\}',
    re.DOTALL,
)

# 6. <invoke name="..."><parameter name="...">...</parameter></invoke>  (Anthropic/Claude style)
_INVOKE_RE = re.compile(
    r'<invoke\s+name=["\']([^"\']+)["\']>(.*?)</invoke>',
    re.DOTALL,
)
_PARAM_RE = re.compile(
    r'<parameter\s+name=["\']([^"\']+)["\']>(.*?)</parameter>',
    re.DOTALL,
)


def _build_tool_calls(raw: list[Any]) -> list[dict[str, Any]] | None:
    result: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict) or "name" not in item:
            continue
        arguments = item.get("arguments", {})
        # If arguments is a JSON string containing an object, parse it
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
                if isinstance(parsed, dict):
                    arguments = json.dumps(parsed, ensure_ascii=False)
                else:
                    arguments = arguments  # keep as-is
            except (json.JSONDecodeError, ValueError):
                repaired = _repair_json(arguments)
                if repaired:
                    try:
                        parsed = json.loads(repaired)
                        if isinstance(parsed, dict):
                            arguments = json.dumps(parsed, ensure_ascii=False)
                        else:
                            arguments = repaired
                    except (json.JSONDecodeError, ValueError):
                        pass
        else:
            arguments = json.dumps(arguments, ensure_ascii=False)

        result.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": str(item["name"]),
                "arguments": arguments,
            },
        })
    return result or None


def _parse_json_array(raw: str) -> list[Any] | None:
    """Parse a JSON array (or bare object) with repair fallback."""
    raw = raw.strip()

    # Handle bare single object: wrap in array
    if raw.startswith("{"):
        raw = _repair_missing_array_bracket(raw)

    arr_start = raw.find("[")
    if arr_start == -1:
        return None
    arr_end = raw.rfind("]")
    if arr_end == -1:
        return None
    json_str = raw[arr_start: arr_end + 1]
    if json_str.strip() == "[]":
        return None

    for attempt in [json_str, _repair_invalid_backslashes(json_str)]:
        try:
            result = json.loads(attempt)
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


def _parse_invoke_block(text: str) -> list[dict[str, Any]] | None:
    """Parse <invoke name="fn"><parameter name="k">v</parameter></invoke> blocks."""
    items = []
    for m in _INVOKE_RE.finditer(text):
        name = m.group(1)
        body = m.group(2)
        args: dict[str, Any] = {}
        for pm in _PARAM_RE.finditer(body):
            param_name = pm.group(1)
            param_value = pm.group(2).strip()
            # Try to parse as JSON, otherwise keep as string
            try:
                args[param_name] = json.loads(param_value)
            except (json.JSONDecodeError, ValueError):
                args[param_name] = param_value
        items.append({"name": name, "arguments": args})
    return _build_tool_calls(items) if items else None


def _parse_ds_segment(segment: str) -> list[dict[str, Any]] | None:
    """
    Parse DeepSeek tool call segment content.
    Each call may be separated by a type/sep line, e.g.:
      type<|tool▁sep|>function
      {"name": "...", "arguments": {...}}
    """
    # Strip type header lines like "type<|tool▁sep|>function"
    sep_re = re.compile(r".*?<[|｜]tool[▁_]sep[|｜]>.*?\n", re.DOTALL)
    cleaned = sep_re.sub("", segment).strip()

    # Try JSON array / object
    arr = _parse_json_array(cleaned)
    if arr is not None:
        return _build_tool_calls(arr)

    # Try individual lines as separate JSON objects
    calls = []
    for line in cleaned.splitlines():
        line = line.strip()
        if not line or not (line.startswith("{") or line.startswith("[")):
            continue
        arr2 = _parse_json_array(line)
        if arr2:
            calls.extend(arr2)
    if calls:
        return _build_tool_calls(calls)
    return None


def _fence_regions(text: str) -> list[tuple[int, int]]:
    """Return (start, end) positions of all ``` … ``` regions in text."""
    return [(m.start(), m.end()) for m in _FENCE_RE.finditer(text)]


def _is_fully_in_fence(match: re.Match, fences: list[tuple[int, int]]) -> bool:
    """True only when the match is entirely contained within a single fence region."""
    ms, me = match.start(), match.end()
    return any(fs <= ms and me <= fe for fs, fe in fences)


def _first_match_prefer_outside(
    pattern: re.Pattern,
    text: str,
    fences: list[tuple[int, int]],
) -> re.Match | None:
    """Return first match NOT fully inside a fence, or None if every match is fenced.

    We deliberately do NOT fall back to fenced matches: a tool call that is
    entirely contained within a code fence is treated as a documentation example
    and is not parsed.  Only matches that start or extend *outside* every fence
    region are considered real tool calls.
    """
    for m in pattern.finditer(text):
        if not _is_fully_in_fence(m, fences):
            return m
    return None


def _parse_tool_calls_from(
    text: str,
    fences: list[tuple[int, int]] | None = None,
) -> list[dict[str, Any]] | None:
    """
    Core parser that operates on a single text string.

    When *fences* is provided, matches that are fully contained within a fence
    region are deprioritised: the parser prefers outside-fence matches first,
    only falling back to fenced ones if nothing else is found.  This lets a
    ``<tool_calls>`` block whose *arguments* happen to contain triple-backtick
    content (e.g. write_file with markdown) still be found and parsed correctly,
    while fenced *example* blocks in the middle of prose are ignored.
    """
    _fences: list[tuple[int, int]] = fences if fences is not None else []

    def first_match(pattern: re.Pattern) -> re.Match | None:
        if _fences:
            return _first_match_prefer_outside(pattern, text, _fences)
        return pattern.search(text)

    def all_matches(pattern: re.Pattern):
        for m in pattern.finditer(text):
            if not _fences or not _is_fully_in_fence(m, _fences):
                yield m

    # 1. Canonical XML format
    m = first_match(_XML_RE)
    if m:
        arr = _parse_json_array(m.group(1))
        if arr is not None:
            result = _build_tool_calls(arr)
            if result:
                return result

    # 2. DeepSeek <|tool▁calls▁begin|>...<|tool▁calls▁end|>
    m = first_match(_DS_BEGIN_RE)
    if m:
        result = _parse_ds_segment(m.group(1))
        if result:
            return result

    # 3. DeepSeek individual <|tool▁call▁begin|>...<|tool▁call▁end|>
    ds_calls: list[dict[str, Any]] = []
    for m in all_matches(_DS_CALL_RE):
        partial = _parse_ds_segment(m.group(1))
        if partial:
            ds_calls.extend(partial)
    if ds_calls:
        return ds_calls

    # 4. Simple <tool_call>...</tool_call>
    simple_calls: list[dict[str, Any]] = []
    for m in all_matches(_SIMPLE_TAG_RE):
        arr = _parse_json_array(m.group(1))
        if arr:
            partial = _build_tool_calls(arr)
            if partial:
                simple_calls.extend(partial)
    if simple_calls:
        return simple_calls

    # 5. {"tool_calls": [...]} bare JSON wrapper
    m = first_match(_BARE_TOOL_CALLS_RE)
    if m:
        raw_obj = m.group(1)
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

    # 6. Bare single-call object: try json.loads first, then regex
    text_stripped = text.strip()
    if text_stripped.startswith("{"):
        try:
            obj = json.loads(text_stripped)
            if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                result = _build_tool_calls([obj])
                if result:
                    return result
        except (json.JSONDecodeError, ValueError):
            pass
    m = first_match(_SINGLE_CALL_RE)
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

    # 7. <invoke name="...">...</invoke>
    # _parse_invoke_block iterates _INVOKE_RE internally; rebuild the call list
    # here so we can apply the same fence-aware skipping.
    invoke_calls: list[dict[str, Any]] = []
    for m in all_matches(_INVOKE_RE):
        partial = _parse_invoke_block(m.group(0))
        if partial:
            invoke_calls.extend(partial)
    if invoke_calls:
        return invoke_calls

    return None


def parse_tool_calls(text: str) -> list[dict[str, Any]] | None:
    """
    Try multiple patterns to extract tool calls from model output.

    Pattern priority (mirrors ds-free-api):
      1. <tool_calls>[...]</tool_calls>            (canonical / our injected format)
      2. <|tool▁calls▁begin|>...<|tool▁calls▁end|>  (DeepSeek native, fuzzy)
      3. <|tool▁call▁begin|>...<|tool▁call▁end|>    (DeepSeek individual, fuzzy)
      4. <tool_call>...</tool_call>                (simple alias)
      5. {"tool_calls": [...]}                     (bare JSON wrapper)
      6. {"name": ..., "arguments": ...}           (single bare object)
      7. <invoke name="...">...</invoke>            (Anthropic/Claude style)

    Strategy:
      Compute fence regions once, then run a single fence-aware parse pass.
      Patterns prefer matches that are NOT fully inside a code fence: this
      correctly handles tool calls whose *arguments* contain triple backticks
      (e.g. write_file with markdown) while still ignoring fenced *example*
      blocks embedded in prose.  If no outside-fence match exists for a given
      pattern, the fenced match is used as a last resort (handles the rare
      model that wraps its entire tool-call block in a code fence).
    """
    fences = _fence_regions(text)
    return _parse_tool_calls_from(text, fences=fences if fences else None)


# ── tool_choice enforcement ────────────────────────────────────────────────────

def enforce_tool_choice(
    tool_calls: list[dict[str, Any]] | None,
    tool_choice: Any,
    tools: list[dict[str, Any]],
    parallel_tool_calls: bool,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """
    Apply tool_choice and parallel_tool_calls constraints to parsed tool_calls.

    Returns:
        (filtered_tool_calls, error_message)
        If error_message is set, the caller should surface it as a diagnostic error.
        filtered_tool_calls may be None (treat as normal text response).
    """
    valid_names = {
        t["function"]["name"]
        for t in tools
        if t.get("type") == "function" and "function" in t
    }

    # tool_choice: "required" — model MUST produce a tool call
    if tool_choice == "required" and not tool_calls:
        return None, (
            "tool_choice is 'required' but the model did not produce a tool call. "
            "The model returned a plain text response instead."
        )

    if not tool_calls:
        return None, None

    # Filter out unknown tool names (tools not in the provided list)
    known = [tc for tc in tool_calls if tc["function"]["name"] in valid_names]
    unknown_names = [tc["function"]["name"] for tc in tool_calls if tc["function"]["name"] not in valid_names]
    if unknown_names and not known:
        # All calls were to unknown tools — treat as no tool call
        return None, (
            f"Model called unknown tool(s) {unknown_names} not present in request.tools. "
            "Returning plain text response."
        )
    tool_calls = known if known else tool_calls

    # tool_choice: named function — only allow that specific tool
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        required_name = (tool_choice.get("function") or {}).get("name", "")
        if required_name:
            named = [tc for tc in tool_calls if tc["function"]["name"] == required_name]
            if named:
                return named[:1], None
            # Named tool not called
            return None, (
                f"tool_choice specifies '{required_name}' but model did not call that tool "
                f"(called: {[tc['function']['name'] for tc in tool_calls]})."
            )

    # parallel_tool_calls: false — keep only first
    if not parallel_tool_calls and len(tool_calls) > 1:
        tool_calls = tool_calls[:1]

    return tool_calls, None


# ── Legacy compatibility ───────────────────────────────────────────────────────

def normalize_legacy_body(body: dict[str, Any]) -> dict[str, Any]:
    """
    Convert old-style OpenAI function calling to the modern tools format:
      - `functions` array → `tools` array
      - `function_call` → `tool_choice`

    The two conversions are independent: a body may carry only `function_call`
    (no `functions`) when the caller already used `tools` but still passes the
    old `function_call` field, or vice-versa.  We handle each field regardless
    of whether the other is present.

    Returns a (possibly new) dict; does not mutate the input.
    """
    has_functions = "functions" in body
    has_function_call = "function_call" in body

    if not has_functions and not has_function_call:
        return body

    body = dict(body)

    # Convert functions[] → tools[]
    if has_functions:
        functions = body.pop("functions", [])
        if "tools" not in body and isinstance(functions, list):
            body["tools"] = [
                {"type": "function", "function": f}
                for f in functions
                if isinstance(f, dict)
            ]

    # Convert function_call → tool_choice (independent of functions presence)
    if has_function_call:
        fc = body.pop("function_call")
        if "tool_choice" not in body:
            if fc in ("none", "auto"):
                body["tool_choice"] = fc
            elif isinstance(fc, dict) and "name" in fc:
                body["tool_choice"] = {"type": "function", "function": {"name": fc["name"]}}
            # else: unknown format, leave default (auto)

    return body


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
    messages: list[dict[str, Any]] | None = None,
    full_text: str | None = None,
) -> dict[str, Any]:
    """Build a non-streaming tool_calls response with real usage stats."""
    from services.protocol.conversation import count_message_text_tokens, count_text_tokens

    prompt_tokens = count_message_text_tokens(messages, model) if messages else 0
    completion_tokens = count_text_tokens(full_text, model) if full_text else 0
    total_tokens = prompt_tokens + completion_tokens

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
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


def stream_tool_calls_chunks(
    model: str,
    tool_calls: list[dict[str, Any]],
    completion_id: str,
    created: int,
    usage: dict[str, int] | None = None,
) -> Iterator[dict[str, Any]]:
    """
    Yield OpenAI-compatible SSE chunks for a tool call response.
    Structured so that OpenAI SDK's accumulate() correctly merges delta.tool_calls.
    The final chunk carries finish_reason='tool_calls' and optional usage.
    """
    def _chunk(delta: dict[str, Any], finish_reason: str | None = None, include_usage: bool = False) -> dict[str, Any]:
        c: dict[str, Any] = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if include_usage and usage:
            c["usage"] = usage
        return c

    # Role chunk
    yield _chunk({"role": "assistant", "content": None})

    # One chunk per tool call: header (name + id), then arguments in pieces
    for idx, tc in enumerate(tool_calls):
        fn = tc["function"]
        # Header chunk: id, type, function.name, empty arguments
        yield _chunk({
            "tool_calls": [{
                "index": idx,
                "id": tc["id"],
                "type": "function",
                "function": {"name": fn["name"], "arguments": ""},
            }]
        })
        # Stream arguments in chunks
        args = fn.get("arguments") or "{}"
        chunk_size = 16
        for start in range(0, len(args), chunk_size):
            yield _chunk({
                "tool_calls": [{
                    "index": idx,
                    "function": {"arguments": args[start: start + chunk_size]},
                }]
            })

    # Final chunk: finish_reason + usage
    yield _chunk({}, "tool_calls", include_usage=True)


def normalize_tool_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert tool_calls / tool-result / legacy function_call messages to a format
    GPT-4 can understand via plain text injection.

    Mirrors ds-free-api's format_assistant / format_tool:
    - assistant.tool_calls  → <tool_calls>[...]</tool_calls> text
    - assistant.function_call (legacy) → same format via tool_calls conversion
    - role=tool             → <tool_results>[...]</tool_results> in a user message
    """
    result: list[dict[str, Any]] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = str(msg.get("role") or "user")

        # Handle legacy function_call in assistant messages
        if role == "assistant" and msg.get("function_call") and not msg.get("tool_calls"):
            fc = msg["function_call"]
            msg = dict(msg)
            raw_args = fc.get("arguments") or "{}"
            try:
                args_obj = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                args_text = json.dumps(args_obj, ensure_ascii=False)
            except Exception:
                args_text = str(raw_args)
            msg["tool_calls"] = [{
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": fc.get("name", "unknown"), "arguments": args_text},
            }]
            msg.pop("function_call", None)

        if role == "assistant" and msg.get("tool_calls"):
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
            # Group consecutive tool messages
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

        # Handle legacy role=function (old OpenAI function result)
        elif role == "function":
            fn_name = str(msg.get("name") or "unknown")
            content = str(msg.get("content") or "")
            result.append({
                "role": "user",
                "content": (
                    f'{TOOL_RESULT_START}[{{"tool_name": "{fn_name}", '
                    f'"output": {json.dumps(content, ensure_ascii=False)}}}]{TOOL_RESULT_END}'
                ),
            })
            i += 1

        else:
            result.append(msg)
            i += 1

    return result
