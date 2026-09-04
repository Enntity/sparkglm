#!/usr/bin/env python3
"""Regression checks for GLM-5.3 chat-template reasoning controls."""

import json
import unittest
from pathlib import Path

from jinja2 import Environment


def _tojson(value, ensure_ascii=False, indent=None, **_kwargs):
    """vLLM's renderer supplies a tojson accepting ensure_ascii; bare Jinja2
    does not. Register the same shape so the template renders standalone."""
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent)


def _environment() -> Environment:
    env = Environment(extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = _tojson
    return env


TEMPLATE = Path(__file__).parents[1] / "files" / "chat_template.jinja"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
    }
]

CONVERSATION = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "Hi. How can I help?"},
    {"role": "user", "content": "and 3+3?"},
]


def render_generation_prompt(**kwargs: object) -> str:
    template = _environment().from_string(TEMPLATE.read_text())
    return template.render(
        messages=[{"role": "user", "content": "hello"}],
        tools=None,
        add_generation_prompt=True,
        **kwargs,
    )


def render_conversation(**kwargs: object) -> str:
    template = _environment().from_string(TEMPLATE.read_text())
    kwargs.setdefault("add_generation_prompt", True)
    return template.render(**kwargs)


class ChatTemplateTests(unittest.TestCase):
    def test_thinking_defaults_on(self) -> None:
        rendered = render_generation_prompt()
        self.assertIn("<|system|>Reasoning Effort: Max", rendered)
        self.assertTrue(rendered.endswith("<|assistant|><think>"), rendered)

    def test_thinking_can_be_disabled(self) -> None:
        rendered = render_generation_prompt(enable_thinking=False)
        # The head line stays: it is what keeps the cached prefix stable across
        # a thinking toggle. Thinking is disabled by the closed block instead.
        self.assertIn("<|system|>Reasoning Effort: Max", rendered)
        self.assertTrue(rendered.endswith("<|assistant|><think></think>"), rendered)

    def test_thinking_alias_matches_parser_behavior(self) -> None:
        rendered = render_generation_prompt(thinking=False)
        self.assertIn("<|system|>Reasoning Effort: Max", rendered)
        self.assertTrue(rendered.endswith("<|assistant|><think></think>"), rendered)

    def test_explicit_thinking_preserves_reasoning_effort(self) -> None:
        rendered = render_generation_prompt(
            enable_thinking=True,
            reasoning_effort="low",
        )
        self.assertIn("<|system|>Reasoning Effort: Low", rendered)
        self.assertTrue(rendered.endswith("<|assistant|><think>"), rendered)

    def test_none_content_is_empty_not_literal_none(self) -> None:
        rendered = render_conversation(
            messages=[{"role": "user", "content": None}],
            tools=None,
            add_generation_prompt=False,
        )
        self.assertEqual(rendered, "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>")

    def test_tool_results_follow_declared_call_order(self) -> None:
        messages = [
            {"role": "user", "content": "run both"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-b",
                        "function": {"name": "second", "arguments": {}},
                    },
                    {
                        "id": "call-a",
                        "function": {"name": "first", "arguments": {}},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call-a", "content": "A-result"},
            {"role": "tool", "tool_call_id": "call-b", "content": "B-result"},
        ]
        rendered = render_conversation(
            messages=messages,
            tools=TOOLS,
            add_generation_prompt=False,
        )
        self.assertLess(rendered.index("B-result"), rendered.index("A-result"))

    def test_zai_early_exit_guards_are_retained(self) -> None:
        source = TEMPLATE.read_text()
        guard = "{%- if not ns_chk.can_sort -%}{%- break -%}{%- endif -%}"
        self.assertEqual(source.count(guard), 3)
        self.assertIn(
            "{%- set ns_chk.can_sort = false -%}\n                    {%- break -%}",
            source,
        )


class PrefixStabilityTests(unittest.TestCase):
    """Toggling thinking must not change the prompt before the final token.

    vLLM chains prefix-cache block hashes forward from token 0, so any
    divergence near the head invalidates the whole prompt. The off-shape must
    therefore be a strict extension of the on-shape.
    """

    def _assert_strict_extension(self, tools) -> None:
        on = render_conversation(
            messages=CONVERSATION, tools=tools, enable_thinking=True
        )
        off = render_conversation(
            messages=CONVERSATION, tools=tools, enable_thinking=False
        )
        self.assertTrue(
            off.startswith(on),
            "thinking-off prompt must extend thinking-on prompt, but they "
            f"diverge at char {len(_common_prefix(on, off))} of {len(on)}",
        )
        self.assertEqual(off[len(on) :], "</think>")

    def test_toggle_is_prefix_stable_without_tools(self) -> None:
        self._assert_strict_extension(None)

    def test_toggle_is_prefix_stable_with_tools(self) -> None:
        # Agent traffic always carries tools; the tools block renders after the
        # reasoning-effort line, so this is the case that actually regressed.
        self._assert_strict_extension(TOOLS)

    def test_effort_levels_share_the_prompt_up_to_the_effort_word(self) -> None:
        low = render_conversation(
            messages=CONVERSATION, tools=TOOLS, enable_thinking=True,
            reasoning_effort="low",
        )
        high = render_conversation(
            messages=CONVERSATION, tools=TOOLS, enable_thinking=True,
            reasoning_effort="high",
        )
        self.assertEqual(len(_common_prefix(low, high)), len("[gMASK]<sop><|system|>Reasoning Effort: "))


def _common_prefix(a: str, b: str) -> str:
    limit = min(len(a), len(b))
    index = 0
    while index < limit and a[index] == b[index]:
        index += 1
    return a[:index]


if __name__ == "__main__":
    unittest.main()
