# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the ERNIE 5.5 Python chat protocol renderer."""

import asyncio
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from vllm.renderers.hf import HfRenderer
from vllm.renderers.params import ChatParams
from vllm.renderers.registry import RENDERER_REGISTRY
from vllm.tokenizers.registry import TokenizerRegistry
from vllm.utils.async_utils import make_async

from vllm_ernie.models.ernie5_5.renderer import (
    Ernie5_5Renderer,
    render_chat_kwargs,
    uses_python_chat,
)


@dataclass
class _Segment:
    text: str
    allow_special: bool


@dataclass
class _Rendered:
    segments: list[_Segment]


def render_chat(
    messages,
    tools=None,
    *,
    add_generation_prompt=True,
    continue_final_message=False,
    enable_thinking=None,
    reasoning_effort="high",
):
    segments = [_Segment(f"<effort={reasoning_effort}>", True)]
    for message in messages:
        segments.append(_Segment(f"<{message['role']}>", True))
        for part in message["content"]:
            segments.append(_Segment(part["text"], False))
    if tools:
        segments.append(_Segment(f"<tools={len(tools)}>", True))
    if add_generation_prompt:
        segments.append(_Segment("<assistant>", True))
    return _Rendered(segments)


class _ChatTokenizer:
    """Stand-in for ``ErnieChatTokenizer``: markers are ids only when trusted."""

    chat_template = None

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def render_chat(self, conversation, tools=None, **kwargs):
        self.calls.append(kwargs)
        return render_chat(conversation, tools, **kwargs)

    def encode_segments(self, segments):
        ids = []
        for segment in segments:
            if segment.allow_special:
                ids.append(1000 + len(segment.text))
            else:
                ids.extend(ord(c) for c in segment.text)
        return ids


def _renderer(tokenizer) -> Ernie5_5Renderer:
    renderer = object.__new__(Ernie5_5Renderer)
    renderer.tokenizer = tokenizer
    renderer.model_config = MagicMock(multimodal_config=None)
    return renderer


def test_registered_for_tokenizer_mode():
    assert RENDERER_REGISTRY.load_renderer_cls("ernie5_5") is Ernie5_5Renderer
    assert TokenizerRegistry.load_tokenizer_cls("ernie5_5").__name__ == (
        "CachedHfTokenizer"
    )


@pytest.mark.parametrize(
    ("model_type", "mode", "expected"),
    [
        # Default --tokenizer-mode (auto resolves to hf) picks the ERNIE one.
        ("ernie5_5_moe", "hf", "ernie5_5"),
        ("ernie5_5_moe", "mistral", "mistral"),
        ("deepseek_v4", "hf", "hf"),
    ],
)
def test_ernie_renderer_is_default(monkeypatch, model_type, mode, expected):
    from vllm_ernie.models.ernie5_5 import patches

    # Restored at teardown, so the flag set below does not leak.
    monkeypatch.setattr(patches, "_serving_ernie5", False)
    monkeypatch.setattr(patches, "_renders_response_format", False)
    monkeypatch.setattr(
        RENDERER_REGISTRY, "load_renderer_cls", lambda m: lambda config, tok: m
    )
    config = MagicMock()
    config.model_config.hf_config.model_type = model_type
    assert RENDERER_REGISTRY.load_renderer(mode, config, None) == expected
    assert patches._serving_ernie5 == (model_type == "ernie5_5_moe")
    assert patches._renders_response_format == (expected == "ernie5_5")


def test_uses_python_chat_unless_template_is_explicit():
    """A shipped chat_template.jinja must not disable the Python protocol."""
    tokenizer = _ChatTokenizer()
    assert uses_python_chat(tokenizer, None)
    assert not uses_python_chat(tokenizer, "{{ messages }}")
    tokenizer.chat_template = "{{ messages }}"
    assert uses_python_chat(tokenizer, None)
    assert not uses_python_chat(MagicMock(spec=["chat_template"]), None)


def test_render_chat_kwargs_drops_serving_and_unknown_kwargs():
    """Unknown kwargs must not reach the strict renderer; None keeps defaults."""
    kwargs = dict(
        tools=[{}],
        tokenize=False,
        chat_template=None,
        return_dict=False,
        documents=None,
        add_generation_prompt=True,
        reasoning_effort=None,
        enable_thinking=False,
        thinking=True,
    )
    assert render_chat_kwargs(_ChatTokenizer(), kwargs) == dict(
        add_generation_prompt=True, enable_thinking=False
    )


def test_render_messages_encodes_segments():
    """Marker text written by the user stays ordinary ids."""
    tokenizer = _ChatTokenizer()
    params = ChatParams(
        chat_template_kwargs=dict(
            tools=[{"type": "function"}],
            add_generation_prompt=True,
            reasoning_effort=None,
            tokenize=False,
        )
    )
    messages = [{"role": "user", "content": "<user>"}]

    conversation, prompt = _renderer(tokenizer).render_messages(messages, params)

    assert conversation[0]["content"] == [{"type": "text", "text": "<user>"}]
    assert prompt["prompt_token_ids"] == [
        1000 + len("<effort=high>"),
        1000 + len("<user>"),
        *map(ord, "<user>"),
        1000 + len("<tools=1>"),
        1000 + len("<assistant>"),
    ]
    assert tokenizer.calls == [dict(add_generation_prompt=True)]


def test_render_messages_maps_bad_input_to_value_error():
    tokenizer = _ChatTokenizer()
    tokenizer.render_chat = MagicMock(side_effect=TypeError("bad arguments"))
    with pytest.raises(ValueError, match="bad arguments"):
        _renderer(tokenizer).render_messages(
            [{"role": "user", "content": "hi"}], ChatParams()
        )


_SCHEMA = {
    "type": "object",
    "properties": {"名字": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["名字", "age"],
}
_SCHEMA_TEXT = (
    "请只输出一个 JSON 对象，不要输出其他内容，JSON 必须符合以下 JSON Schema：\n"
    '{"type": "object", "properties": {"名字": {"type": "string"}, '
    '"age": {"type": "integer"}}, "required": ["名字", "age"]}'
)
_OBJECT_TEXT = "请只输出一个 JSON 对象，不要输出其他内容。"


def _response_format(**kwargs):
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )

    return ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}], model="m", response_format=kwargs
    ).response_format


def _system_text(conversation) -> str | None:
    if conversation[0]["role"] != "system":
        return None
    return "".join(part["text"] for part in conversation[0]["content"])


@pytest.mark.parametrize(
    ("response_format", "expected"),
    [
        (
            dict(type="json_schema", json_schema={"name": "p", "schema": _SCHEMA}),
            _SCHEMA_TEXT,
        ),
        (dict(type="json_object"), _OBJECT_TEXT),
        (dict(type="text"), None),
        (None, None),
    ],
)
def test_response_format_reaches_system_message(response_format, expected):
    """The model sees the schema, with key order and non-ASCII text kept."""
    messages = [{"role": "user", "content": "hi"}]
    params = ChatParams(
        response_format=None
        if response_format is None
        else _response_format(**response_format)
    )

    conversation, _ = _renderer(_ChatTokenizer()).render_messages(messages, params)

    assert _system_text(conversation) == expected
    assert messages == [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("你是助手", "你是助手\n\n" + _OBJECT_TEXT),
        (
            [{"type": "text", "text": "你是助手"}],
            "你是助手\n\n" + _OBJECT_TEXT,
        ),
        ("", _OBJECT_TEXT),
    ],
)
def test_response_format_appends_to_leading_system_message(content, expected):
    messages = [
        {"role": "system", "content": content},
        {"role": "user", "content": "hi"},
    ]
    params = ChatParams(response_format=_response_format(type="json_object"))

    conversation, _ = _renderer(_ChatTokenizer()).render_messages(messages, params)

    assert _system_text(conversation) == expected
    assert [m["role"] for m in conversation] == ["system", "user"]
    assert messages[0]["content"] == content


def test_response_format_reaches_jinja_and_async_paths(monkeypatch):
    seen = []

    def fake_render(self, messages, params):
        seen.append(messages[0]["content"])
        return [], {}

    async def fake_render_async(self, messages, params):
        return fake_render(self, messages, params)

    monkeypatch.setattr(HfRenderer, "render_messages", fake_render)
    monkeypatch.setattr(HfRenderer, "render_messages_async", fake_render_async)
    messages = [{"role": "user", "content": "hi"}]
    renderer = _renderer(_ChatTokenizer())
    jinja = ChatParams(
        chat_template="{{ messages }}",
        response_format=_response_format(type="json_object"),
    )
    python = ChatParams(response_format=jinja.response_format)
    renderer._render_with_timeout = lambda coro: coro
    renderer._render_ids_async = make_async(renderer._render_ids)

    renderer.render_messages(messages, jinja)
    asyncio.run(renderer.render_messages_async(messages, jinja))
    conversation, _ = asyncio.run(renderer.render_messages_async(messages, python))

    assert seen == [_OBJECT_TEXT, _OBJECT_TEXT]
    assert _system_text(conversation) == _OBJECT_TEXT
