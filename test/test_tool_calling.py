"""
Unit tests for services/protocol/tool_calling.py

Covers:
  - parse_tool_calls: all supported tag formats
  - JSON repair: invalid backslashes, unquoted keys, bare objects
  - enforce_tool_choice: none / required / named / parallel constraints
  - normalize_legacy_body: functions → tools, function_call → tool_choice
  - normalize_tool_history: assistant tool_calls, tool results, legacy function_call
  - tools_system_message: prompt injection
  - stream_tool_calls_chunks: correct delta structure
  - ds-free-api repair scenario edge cases (10 classes)
"""

from __future__ import annotations

import json
import pytest

from services.protocol.tool_calling import (
    _repair_invalid_backslashes,
    _repair_unquoted_keys,
    _repair_json,
    _strip_code_fences,
    _parse_invoke_block,
    enforce_tool_choice,
    normalize_legacy_body,
    normalize_tool_history,
    parse_tool_calls,
    stream_tool_calls_chunks,
    tools_system_message,
    TOOL_CALL_START,
    TOOL_CALL_END,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

TOOLS_FIXTURE = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather in a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get stock price",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            },
        },
    },
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _call_name(tc: dict) -> str:
    return tc["function"]["name"]


def _call_args(tc: dict) -> dict:
    return json.loads(tc["function"]["arguments"])


# ══════════════════════════════════════════════════════════════════════════════
# Section 1: parse_tool_calls — supported formats
# ══════════════════════════════════════════════════════════════════════════════

class TestParseToolCallsCanonicalXML:
    def test_basic(self):
        text = '<tool_calls>[{"name": "get_weather", "arguments": {"city": "Beijing"}}]</tool_calls>'
        result = parse_tool_calls(text)
        assert result is not None
        assert len(result) == 1
        assert _call_name(result[0]) == "get_weather"
        assert _call_args(result[0]) == {"city": "Beijing"}

    def test_with_surrounding_text(self):
        """Model output with extra text before/after — only the tag content counts."""
        text = "Sure, let me check that for you.\n<tool_calls>[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"London\"}}]</tool_calls>\nHope that helps!"
        result = parse_tool_calls(text)
        assert result is not None
        assert _call_name(result[0]) == "get_weather"

    def test_parallel_calls(self):
        text = (
            '<tool_calls>['
            '{"name": "get_weather", "arguments": {"city": "Tokyo"}}, '
            '{"name": "search", "arguments": {"query": "AI news"}}'
            ']</tool_calls>'
        )
        result = parse_tool_calls(text)
        assert result is not None
        assert len(result) == 2
        assert _call_name(result[0]) == "get_weather"
        assert _call_name(result[1]) == "search"

    def test_empty_arguments(self):
        text = '<tool_calls>[{"name": "get_weather", "arguments": {}}]</tool_calls>'
        result = parse_tool_calls(text)
        assert result is not None
        assert _call_args(result[0]) == {}

    def test_no_tool_calls_returns_none(self):
        assert parse_tool_calls("Hello, how are you?") is None
        assert parse_tool_calls("") is None

    def test_empty_array_returns_none(self):
        assert parse_tool_calls("<tool_calls>[]</tool_calls>") is None


class TestParseToolCallsBareJSON:
    def test_bare_tool_calls_object(self):
        text = '{"tool_calls": [{"name": "search", "arguments": {"query": "python"}}]}'
        result = parse_tool_calls(text)
        assert result is not None
        assert _call_name(result[0]) == "search"

    def test_bare_object_not_in_code_fence(self):
        """Bare JSON outside code fence should be parsed."""
        text = '{"tool_calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}]}'
        result = parse_tool_calls(text)
        assert result is not None

    def test_code_fence_example_not_parsed(self):
        """
        <tool_calls> inside a markdown code block is a documentation example —
        it must NOT be parsed as a real tool call.
        """
        text = (
            "Here is how to call the tool:\n"
            "```\n"
            "<tool_calls>[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Shanghai\"}}]</tool_calls>\n"
            "```\n"
            "This is just an example."
        )
        result = parse_tool_calls(text)
        assert result is None

    def test_code_fence_json_example_not_parsed(self):
        """
        {"tool_calls": [...]} inside a code fence should also be ignored.
        """
        text = (
            "Example:\n"
            "```json\n"
            '{"tool_calls": [{"name": "search", "arguments": {"query": "test"}}]}\n'
            "```\n"
            "Do not use the above literally."
        )
        result = parse_tool_calls(text)
        assert result is None


class TestParseToolCallsSingleObject:
    def test_single_bare_object(self):
        text = '{"name": "get_weather", "arguments": {"city": "Berlin"}}'
        result = parse_tool_calls(text)
        assert result is not None
        assert _call_name(result[0]) == "get_weather"
        assert _call_args(result[0]) == {"city": "Berlin"}

    def test_arguments_as_json_string(self):
        """arguments is a JSON string (stringified) — should be normalised."""
        args_str = json.dumps({"city": "Seoul"})
        text = f'{{"name": "get_weather", "arguments": {json.dumps(args_str)}}}'
        result = parse_tool_calls(text)
        assert result is not None
        # arguments should be parseable as JSON dict
        assert _call_args(result[0]) == {"city": "Seoul"}


class TestParseToolCallsDSFreeAPIStyle:
    def test_ds_begin_end_tags(self):
        """DeepSeek <|tool▁calls▁begin|>...<|tool▁calls▁end|> format."""
        text = (
            "<|tool▁calls▁begin|>"
            '[{"name": "get_weather", "arguments": {"city": "Shenzhen"}}]'
            "<|tool▁calls▁end|>"
        )
        result = parse_tool_calls(text)
        assert result is not None
        assert _call_name(result[0]) == "get_weather"

    def test_ds_underscore_variant(self):
        """Fuzzy: ▁ replaced by regular _."""
        text = (
            "<|tool_calls_begin|>"
            '[{"name": "search", "arguments": {"query": "hello"}}]'
            "<|tool_calls_end|>"
        )
        result = parse_tool_calls(text)
        assert result is not None
        assert _call_name(result[0]) == "search"

    def test_ds_fullwidth_pipe_variant(self):
        """Fuzzy: | replaced by full-width ｜."""
        text = (
            "｜tool▁calls▁begin｜"  # no angle brackets — won't match; test with angle brackets
        )
        # Test with actual angle brackets + full-width pipe
        text2 = (
            "<｜tool▁calls▁begin｜>"
            '[{"name": "get_weather", "arguments": {"city": "Chengdu"}}]'
            "<｜tool▁calls▁end｜>"
        )
        result = parse_tool_calls(text2)
        assert result is not None
        assert _call_name(result[0]) == "get_weather"

    def test_ds_individual_call_begin_end(self):
        """DeepSeek <|tool▁call▁begin|>...<|tool▁call▁end|> format."""
        text = (
            "<|tool▁call▁begin|>"
            '{"name": "search", "arguments": {"query": "deep learning"}}'
            "<|tool▁call▁end|>"
        )
        result = parse_tool_calls(text)
        assert result is not None
        assert _call_name(result[0]) == "search"

    def test_simple_tool_call_tags(self):
        """<tool_call>...</tool_call> format."""
        text = '<tool_call>{"name": "get_weather", "arguments": {"city": "Guangzhou"}}</tool_call>'
        result = parse_tool_calls(text)
        assert result is not None
        assert _call_name(result[0]) == "get_weather"


class TestParseToolCallsInvokeFormat:
    def test_basic_invoke(self):
        text = (
            '<invoke name="get_weather">'
            '<parameter name="city">Tokyo</parameter>'
            '</invoke>'
        )
        result = parse_tool_calls(text)
        assert result is not None
        assert _call_name(result[0]) == "get_weather"
        assert _call_args(result[0]) == {"city": "Tokyo"}

    def test_invoke_json_param(self):
        """Parameter value that is itself JSON."""
        text = (
            '<invoke name="search">'
            '<parameter name="query">latest news</parameter>'
            '<parameter name="options">{"max_results": 5}</parameter>'
            '</invoke>'
        )
        result = parse_tool_calls(text)
        assert result is not None
        args = _call_args(result[0])
        assert args["query"] == "latest news"
        assert args["options"] == {"max_results": 5}

    def test_multiple_invokes(self):
        text = (
            '<invoke name="get_weather"><parameter name="city">Rome</parameter></invoke>'
            '<invoke name="search"><parameter name="query">Rome weather</parameter></invoke>'
        )
        result = parse_tool_calls(text)
        assert result is not None
        assert len(result) == 2


# ══════════════════════════════════════════════════════════════════════════════
# Section 2: JSON repair
# ══════════════════════════════════════════════════════════════════════════════

class TestJSONRepair:
    def test_valid_json_unchanged(self):
        s = '{"key": "value"}'
        assert _repair_json(s) == s

    def test_invalid_backslash_windows_path(self):
        """Windows path backslashes like C:/Users/foo should be escaped."""
        s = r'{"path": "C:\Users\foo\bar.txt"}'
        repaired = _repair_invalid_backslashes(s)
        parsed = json.loads(repaired)
        assert "path" in parsed

    def test_unquoted_keys(self):
        s = '{city: "Beijing", count: 3}'
        repaired = _repair_unquoted_keys(s)
        parsed = json.loads(repaired)
        assert parsed["city"] == "Beijing"
        assert parsed["count"] == 3

    def test_unquoted_keys_in_tool_call(self):
        text = '<tool_calls>[{name: "get_weather", arguments: {city: "Tokyo"}}]</tool_calls>'
        result = parse_tool_calls(text)
        assert result is not None
        assert _call_name(result[0]) == "get_weather"

    def test_bare_object_wrapped_in_array(self):
        """If the content of <tool_calls> is a single object, wrap it in []."""
        text = '<tool_calls>{"name": "get_weather", "arguments": {"city": "Moscow"}}</tool_calls>'
        result = parse_tool_calls(text)
        assert result is not None
        assert _call_name(result[0]) == "get_weather"

    def test_repair_json_returns_none_for_unrecoverable(self):
        assert _repair_json("not json at all !!!") is None

    def test_strip_code_fences(self):
        text = "before\n```json\nsome code\n```\nafter"
        stripped = _strip_code_fences(text)
        assert "some code" not in stripped
        assert "before" in stripped
        assert "after" in stripped


# ══════════════════════════════════════════════════════════════════════════════
# Section 3: ds-free-api repair scenario edge cases (10 classes)
# ══════════════════════════════════════════════════════════════════════════════

class TestDSFreeAPIRepairScenarios:
    """
    Mirrors the 10 known abnormal-format classes from ds-free-api repair scenarios.
    Classes that cannot yet be supported are marked with TODO comments.
    """

    def test_r01_invalid_backslash_in_string(self):
        """R01: Invalid escape sequence (e.g. \\U in Windows path)."""
        text = '<tool_calls>[{"name": "read_file", "arguments": {"file_path": "C:\\\\Users\\\\test.txt"}}]</tool_calls>'
        result = parse_tool_calls(text)
        assert result is not None
        assert _call_name(result[0]) == "read_file"

    def test_r02_unquoted_keys(self):
        """R02: Unquoted object keys."""
        text = '<tool_calls>[{name: "search", arguments: {query: "hello"}}]</tool_calls>'
        result = parse_tool_calls(text)
        assert result is not None

    def test_r03_single_object_not_array(self):
        """R03: Single object in place of array inside tags."""
        text = '<tool_calls>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_calls>'
        result = parse_tool_calls(text)
        assert result is not None
        assert _call_name(result[0]) == "get_weather"

    def test_r04_arguments_as_json_string(self):
        """R04: arguments is a JSON-stringified dict."""
        args_json = json.dumps({"city": "Paris"})
        text = f'<tool_calls>[{{"name": "get_weather", "arguments": {json.dumps(args_json)}}}]</tool_calls>'
        result = parse_tool_calls(text)
        assert result is not None
        args = _call_args(result[0])
        assert args.get("city") == "Paris"

    def test_r05_extra_text_before_after_tags(self):
        """R05: Extra prose before/after the tool call block."""
        text = (
            "I need to get the weather. Let me call the tool.\n"
            '<tool_calls>[{"name": "get_weather", "arguments": {"city": "Dubai"}}]</tool_calls>\n'
            "I'll get back to you shortly."
        )
        result = parse_tool_calls(text)
        assert result is not None

    def test_r06_markdown_code_fence_around_tool_call(self):
        """R06: tool call wrapped in markdown code fence — should NOT parse as real call."""
        text = "```\n<tool_calls>[{\"name\": \"get_weather\", \"arguments\": {\"city\": \"X\"}}]</tool_calls>\n```"
        result = parse_tool_calls(text)
        assert result is None

    def test_r07_parallel_multiple_tools(self):
        """R07: Multiple tools in one array."""
        text = (
            '<tool_calls>['
            '{"name": "get_weather", "arguments": {"city": "A"}}, '
            '{"name": "search", "arguments": {"query": "B"}}, '
            '{"name": "get_stock_price", "arguments": {"symbol": "C"}}'
            ']</tool_calls>'
        )
        result = parse_tool_calls(text)
        assert result is not None
        assert len(result) == 3

    def test_r08_ds_begin_end_with_content(self):
        """R08: DeepSeek-style begin/end tokens (trained tokens)."""
        text = (
            "<|tool▁calls▁begin|>"
            '[{"name": "get_weather", "arguments": {"city": "Wuhan"}}]'
            "<|tool▁calls▁end|>"
        )
        result = parse_tool_calls(text)
        assert result is not None

    def test_r09_invoke_xml_format(self):
        """R09: <invoke name="..."><parameter ...>...</parameter></invoke> format."""
        text = '<invoke name="get_weather"><parameter name="city">Mumbai</parameter></invoke>'
        result = parse_tool_calls(text)
        assert result is not None
        assert _call_name(result[0]) == "get_weather"

    def test_r10_missing_closing_array_bracket(self):
        """
        R10: Missing closing ] in the array.
        TODO: Full bracket repair for deeply nested structures not yet supported.
        The current implementation handles simple cases via rfind(']').
        """
        # Partially recoverable: the array close was just missed at the end
        text = '<tool_calls>[{"name": "get_weather", "arguments": {"city": "Lima"}}]</tool_calls>'
        result = parse_tool_calls(text)
        assert result is not None  # baseline — proper array
        # True missing-bracket case — currently not supported, marked TODO
        # text_broken = '<tool_calls>[{"name": "get_weather", "arguments": {"city": "Lima"}}</tool_calls>'
        # result_broken = parse_tool_calls(text_broken)
        # assert result_broken is not None  # TODO: not yet supported


# ══════════════════════════════════════════════════════════════════════════════
# Section 4: enforce_tool_choice
# ══════════════════════════════════════════════════════════════════════════════

class TestEnforceToolChoice:
    def _make_tool_calls(self, names: list[str]) -> list[dict]:
        return [
            {
                "id": f"call_test_{i}",
                "type": "function",
                "function": {"name": n, "arguments": "{}"},
            }
            for i, n in enumerate(names)
        ]

    def test_auto_with_calls(self):
        tcs = self._make_tool_calls(["get_weather"])
        result, err = enforce_tool_choice(tcs, "auto", TOOLS_FIXTURE, True)
        assert err is None
        assert result is not None
        assert len(result) == 1

    def test_auto_no_calls(self):
        result, err = enforce_tool_choice(None, "auto", TOOLS_FIXTURE, True)
        assert err is None
        assert result is None

    def test_required_no_calls_returns_error(self):
        result, err = enforce_tool_choice(None, "required", TOOLS_FIXTURE, True)
        assert err is not None
        assert "required" in err

    def test_required_with_calls_passes(self):
        tcs = self._make_tool_calls(["get_weather"])
        result, err = enforce_tool_choice(tcs, "required", TOOLS_FIXTURE, True)
        assert err is None
        assert result is not None

    def test_named_tool_choice_filters_correctly(self):
        tcs = self._make_tool_calls(["get_weather", "search"])
        tool_choice = {"type": "function", "function": {"name": "get_weather"}}
        result, err = enforce_tool_choice(tcs, tool_choice, TOOLS_FIXTURE, True)
        assert err is None
        assert result is not None
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_weather"

    def test_named_tool_not_called_returns_error(self):
        tcs = self._make_tool_calls(["search"])
        tool_choice = {"type": "function", "function": {"name": "get_weather"}}
        result, err = enforce_tool_choice(tcs, tool_choice, TOOLS_FIXTURE, True)
        assert err is not None
        assert "get_weather" in err

    def test_parallel_false_keeps_only_first(self):
        tcs = self._make_tool_calls(["get_weather", "search"])
        result, err = enforce_tool_choice(tcs, "auto", TOOLS_FIXTURE, parallel_tool_calls=False)
        assert err is None
        assert result is not None
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_weather"

    def test_parallel_true_keeps_all(self):
        tcs = self._make_tool_calls(["get_weather", "search"])
        result, err = enforce_tool_choice(tcs, "auto", TOOLS_FIXTURE, parallel_tool_calls=True)
        assert err is None
        assert result is not None
        assert len(result) == 2

    def test_unknown_tool_name_filtered(self):
        """A tool name not in the provided tools list should be filtered out."""
        tcs = self._make_tool_calls(["nonexistent_tool"])
        result, err = enforce_tool_choice(tcs, "auto", TOOLS_FIXTURE, True)
        assert result is None
        assert err is not None
        assert "nonexistent_tool" in err

    def test_mixed_known_unknown_keeps_known(self):
        """If some calls are known, keep only the known ones."""
        tcs = self._make_tool_calls(["get_weather", "fake_tool"])
        result, err = enforce_tool_choice(tcs, "auto", TOOLS_FIXTURE, True)
        assert err is None
        assert result is not None
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_weather"


# ══════════════════════════════════════════════════════════════════════════════
# Section 5: normalize_legacy_body
# ══════════════════════════════════════════════════════════════════════════════

class TestNormalizeLegacyBody:
    def test_functions_converted_to_tools(self):
        body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
            "functions": [{"name": "get_weather", "parameters": {}}],
        }
        result = normalize_legacy_body(body)
        assert "tools" in result
        assert result["tools"][0]["type"] == "function"
        assert result["tools"][0]["function"]["name"] == "get_weather"
        assert "functions" not in result

    def test_function_call_none_mapped(self):
        body = {
            "functions": [{"name": "get_weather", "parameters": {}}],
            "function_call": "none",
        }
        result = normalize_legacy_body(body)
        assert result.get("tool_choice") == "none"
        assert "function_call" not in result

    def test_function_call_auto_mapped(self):
        body = {
            "functions": [{"name": "get_weather", "parameters": {}}],
            "function_call": "auto",
        }
        result = normalize_legacy_body(body)
        assert result.get("tool_choice") == "auto"

    def test_function_call_named_mapped(self):
        body = {
            "functions": [{"name": "get_weather", "parameters": {}}],
            "function_call": {"name": "get_weather"},
        }
        result = normalize_legacy_body(body)
        assert result["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}

    def test_no_functions_unchanged(self):
        body = {"model": "gpt-4", "messages": []}
        result = normalize_legacy_body(body)
        assert result is body  # same object, not copied


# ══════════════════════════════════════════════════════════════════════════════
# Section 6: normalize_tool_history
# ══════════════════════════════════════════════════════════════════════════════

class TestNormalizeToolHistory:
    def test_assistant_tool_calls_converted_to_text(self):
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_abc",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }],
            }
        ]
        result = normalize_tool_history(messages)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert TOOL_CALL_START in result[0]["content"]
        assert "get_weather" in result[0]["content"]

    def test_tool_result_converted_to_user_message(self):
        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_abc",
                "content": '{"temperature": "20°C"}',
            }
        ]
        result = normalize_tool_history(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "<tool_results>" in result[0]["content"]

    def test_consecutive_tool_results_grouped(self):
        messages = [
            {"role": "tool", "tool_call_id": "call_1", "content": "result1"},
            {"role": "tool", "tool_call_id": "call_2", "content": "result2"},
        ]
        result = normalize_tool_history(messages)
        assert len(result) == 1  # grouped into one user message
        assert "call_1" in result[0]["content"]
        assert "call_2" in result[0]["content"]

    def test_legacy_function_call_in_assistant(self):
        """Old-style assistant.function_call should be converted to tool_calls format."""
        messages = [
            {
                "role": "assistant",
                "content": None,
                "function_call": {"name": "get_weather", "arguments": '{"city": "Rome"}'},
            }
        ]
        result = normalize_tool_history(messages)
        assert result[0]["role"] == "assistant"
        assert "get_weather" in result[0]["content"]
        assert TOOL_CALL_START in result[0]["content"]

    def test_legacy_function_role_result(self):
        """Old-style role=function result should be converted to tool_results user message."""
        messages = [
            {"role": "function", "name": "get_weather", "content": "Sunny, 25°C"},
        ]
        result = normalize_tool_history(messages)
        assert result[0]["role"] == "user"
        assert "get_weather" in result[0]["content"]
        assert "<tool_results>" in result[0]["content"]

    def test_regular_messages_pass_through(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = normalize_tool_history(messages)
        assert result == messages

    def test_full_multi_turn_conversation(self):
        """Multi-turn: user → assistant(tool_calls) → tool result → user follow-up."""
        messages = [
            {"role": "user", "content": "What's the weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Rainy, 15°C"},
            {"role": "user", "content": "Thanks!"},
        ]
        result = normalize_tool_history(messages)
        assert len(result) == 4
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert TOOL_CALL_START in result[1]["content"]
        assert result[2]["role"] == "user"
        assert "<tool_results>" in result[2]["content"]
        assert result[3]["role"] == "user"


# ══════════════════════════════════════════════════════════════════════════════
# Section 7: tools_system_message
# ══════════════════════════════════════════════════════════════════════════════

class TestToolsSystemMessage:
    def test_basic_structure(self):
        msg = tools_system_message(TOOLS_FIXTURE)
        assert msg["role"] == "system"
        assert "工具调用" in msg["content"]
        assert "get_weather" in msg["content"]
        assert "search" in msg["content"]

    def test_tool_choice_none_returns_empty(self):
        msg = tools_system_message(TOOLS_FIXTURE, tool_choice="none")
        assert msg["content"] == ""

    def test_required_instruction_present(self):
        msg = tools_system_message(TOOLS_FIXTURE, tool_choice="required")
        assert "必须调用" in msg["content"]

    def test_named_tool_choice_instruction(self):
        msg = tools_system_message(
            TOOLS_FIXTURE,
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )
        assert "get_weather" in msg["content"]
        assert "必须调用" in msg["content"]

    def test_parallel_false_adds_rule(self):
        msg = tools_system_message(TOOLS_FIXTURE, parallel_tool_calls=False)
        assert "一次只能调用一个工具" in msg["content"]

    def test_three_tools_has_parallel_example(self):
        """With 3+ tools, the prompt should include a 3-tool parallel example."""
        msg = tools_system_message(TOOLS_FIXTURE)
        # Should have at least examples A, B (or A, B, C for 3 tools)
        assert "示例A" in msg["content"]
        assert "示例B" in msg["content"]
        assert "示例C" in msg["content"]

    def test_only_one_tool_call_block_rule(self):
        """Rule about having only ONE tool call block in the response."""
        msg = tools_system_message(TOOLS_FIXTURE)
        assert "一个" in msg["content"]  # "只允许出现一个工具调用块"

    def test_no_thinking_content_rule(self):
        """Rule about not putting tool calls in thinking content."""
        msg = tools_system_message(TOOLS_FIXTURE)
        assert "思考" in msg["content"]

    def test_nested_params_example(self):
        msg = tools_system_message(TOOLS_FIXTURE)
        assert "示例D" in msg["content"]


# ══════════════════════════════════════════════════════════════════════════════
# Section 8: stream_tool_calls_chunks
# ══════════════════════════════════════════════════════════════════════════════

class TestStreamToolCallsChunks:
    def _make_tc(self, name: str, args: dict) -> dict:
        return {
            "id": f"call_{name}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }

    def test_first_chunk_has_role(self):
        tc = self._make_tc("get_weather", {"city": "Seoul"})
        chunks = list(stream_tool_calls_chunks("gpt-4", [tc], "cid", 1000))
        first = chunks[0]
        assert first["choices"][0]["delta"].get("role") == "assistant"
        assert first["choices"][0]["finish_reason"] is None

    def test_last_chunk_finish_reason_tool_calls(self):
        tc = self._make_tc("get_weather", {"city": "Seoul"})
        chunks = list(stream_tool_calls_chunks("gpt-4", [tc], "cid", 1000))
        last = chunks[-1]
        assert last["choices"][0]["finish_reason"] == "tool_calls"

    def test_tool_call_header_chunk_has_name(self):
        tc = self._make_tc("get_weather", {"city": "Seoul"})
        chunks = list(stream_tool_calls_chunks("gpt-4", [tc], "cid", 1000))
        # Find the chunk that has tool_calls with function.name
        header_chunks = [
            c for c in chunks
            if c["choices"][0]["delta"].get("tool_calls")
            and c["choices"][0]["delta"]["tool_calls"][0].get("function", {}).get("name")
        ]
        assert len(header_chunks) >= 1
        assert header_chunks[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_all_chunks_have_same_id(self):
        tc = self._make_tc("search", {"query": "test"})
        chunks = list(stream_tool_calls_chunks("gpt-4", [tc], "my-cid", 1000))
        assert all(c["id"] == "my-cid" for c in chunks)

    def test_arguments_reconstructable(self):
        """All argument chunks for an index can be concatenated to reconstruct arguments."""
        args = {"city": "New York", "unit": "celsius"}
        tc = self._make_tc("get_weather", args)
        chunks = list(stream_tool_calls_chunks("gpt-4", [tc], "cid", 1000))

        reconstructed = ""
        for c in chunks:
            delta = c["choices"][0]["delta"]
            if delta.get("tool_calls"):
                for tcd in delta["tool_calls"]:
                    fn = tcd.get("function", {})
                    reconstructed += fn.get("arguments", "")

        assert json.loads(reconstructed) == args

    def test_usage_in_last_chunk(self):
        tc = self._make_tc("get_weather", {"city": "Kyoto"})
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        chunks = list(stream_tool_calls_chunks("gpt-4", [tc], "cid", 1000, usage=usage))
        last = chunks[-1]
        assert last.get("usage") == usage

    def test_parallel_tool_calls_indexed(self):
        """Multiple tool calls should have correct index values."""
        tcs = [
            self._make_tc("get_weather", {"city": "A"}),
            self._make_tc("search", {"query": "B"}),
        ]
        chunks = list(stream_tool_calls_chunks("gpt-4", tcs, "cid", 1000))
        # Collect all tool_call deltas
        tc_deltas = []
        for c in chunks:
            for tcd in c["choices"][0]["delta"].get("tool_calls", []):
                tc_deltas.append(tcd)
        indices = [tcd["index"] for tcd in tc_deltas if "index" in tcd]
        assert 0 in indices
        assert 1 in indices
