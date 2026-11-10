# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ERNIE 5.5 chat renderer.

Newer ERNIE 5.5 checkpoints ship a Python chat protocol (``ErnieChatTokenizer``
with ``render_chat`` / ``encode_segments``) instead of a Jinja template.
Rendering to text and re-tokenizing would turn marker strings quoted in user or
tool text into control tokens, so such prompts are encoded segment by segment:
only renderer-emitted markers become added-token ids.

Tokenizers with a Jinja template, or requests with an explicit chat template,
keep the ``HfRenderer`` path.
"""

import functools
import inspect
from typing import Any

from vllm.logger import init_logger
from vllm.renderers.chat_utils import (
    ChatCompletionMessageParam,
    ConversationMessage,
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.renderers.hf import HfRenderer
from vllm.renderers.inputs import DictPrompt
from vllm.renderers.inputs.preprocess import parse_dec_only_prompt
from vllm.renderers.params import ChatParams
from vllm.utils.async_utils import make_async

logger = init_logger(__name__)

# Passed by vLLM to every HF chat template, not chat protocol options.
_SERVING_KWARGS = frozenset({"chat_template", "return_dict", "tokenize", "documents"})


def uses_python_chat(tokenizer: Any, chat_template: str | None) -> bool:
    """Whether to render with the tokenizer's Python chat protocol."""
    return (
        chat_template is None
        and getattr(tokenizer, "chat_template", None) is None
        and callable(getattr(tokenizer, "render_chat", None))
        and callable(getattr(tokenizer, "encode_segments", None))
    )


@functools.cache
def _accepted_kwargs(tokenizer_cls: type) -> frozenset[str] | None:
    """Keyword names accepted by the protocol's ``render_chat``.

    ``ErnieChatTokenizer.render_chat`` forwards ``**kwargs`` to the
    module-level ``render_chat`` of its remote-code module, whose signature is
    explicit. ``None`` means the accepted names are unknown.
    """
    module = inspect.getmodule(tokenizer_cls.render_chat)
    for fn in (getattr(module, "render_chat", None), tokenizer_cls.render_chat):
        if not callable(fn):
            continue
        try:
            params = inspect.signature(fn).parameters.values()
        except (TypeError, ValueError):
            continue
        if any(p.kind is p.VAR_KEYWORD for p in params):
            continue
        return frozenset(
            p.name
            for p in params
            if p.kind in (p.KEYWORD_ONLY, p.POSITIONAL_OR_KEYWORD)
        )
    return None


def render_chat_kwargs(tokenizer: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop the kwargs ``render_chat`` would reject.

    Jinja silently ignores variables a template never reads; the Python
    renderer raises instead. ``None`` values are dropped too, so that vLLM's
    request defaults (e.g. ``reasoning_effort=None``) keep the protocol's own
    defaults.
    """
    accepted = _accepted_kwargs(type(tokenizer))
    kept: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key == "tools" or value is None or key in _SERVING_KWARGS:
            continue
        if accepted is not None and key not in accepted:
            logger.warning_once(
                "Ignoring chat_template_kwargs[%r]: not accepted by %s.render_chat",
                key,
                type(tokenizer).__name__,
            )
            continue
        kept[key] = value
    return kept


class Ernie5_5Renderer(HfRenderer):
    """``HfRenderer`` that renders the ERNIE Python chat protocol to ids."""

    def __init__(self, config, tokenizer) -> None:
        super().__init__(config, tokenizer)
        self._render_ids_async = make_async(self._render_ids, executor=self._executor)

    def _render_ids(
        self,
        conversation: list[ConversationMessage],
        params: ChatParams,
    ) -> list[int]:
        tokenizer = self.get_tokenizer()
        kwargs = params.chat_template_kwargs
        try:
            rendered = tokenizer.render_chat(
                conversation,
                tools=kwargs.get("tools"),
                **render_chat_kwargs(tokenizer, kwargs),
            )
            return list(tokenizer.encode_segments(rendered.segments))
        except ValueError:
            raise
        except (TypeError, KeyError) as e:
            # Malformed request content; surface as a request error.
            raise ValueError(str(e)) from e

    def _to_prompt(self, token_ids: list[int], mm_data, mm_uuids) -> DictPrompt:
        prompt = parse_dec_only_prompt(token_ids)
        if mm_data is not None:
            prompt["multi_modal_data"] = mm_data
        if mm_uuids is not None:
            prompt["multi_modal_uuids"] = mm_uuids
        return prompt

    def render_messages(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        if not uses_python_chat(self.tokenizer, params.chat_template):
            return super().render_messages(messages, params)

        conversation, mm_data, mm_uuids = parse_chat_messages(
            messages,
            self.model_config,
            content_format="openai",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )
        token_ids = self._render_ids(conversation, params)
        return conversation, self._to_prompt(token_ids, mm_data, mm_uuids)

    async def render_messages_async(
        self,
        messages: list[ChatCompletionMessageParam],
        params: ChatParams,
    ) -> tuple[list[ConversationMessage], DictPrompt]:
        if not uses_python_chat(self.tokenizer, params.chat_template):
            return await super().render_messages_async(messages, params)

        conversation, mm_data, mm_uuids = await parse_chat_messages_async(
            messages,
            self.model_config,
            content_format="openai",
            media_io_kwargs=params.media_io_kwargs,
            mm_processor_kwargs=params.mm_processor_kwargs,
        )
        token_ids = await self._render_with_timeout(
            self._render_ids_async(conversation, params)
        )
        return conversation, self._to_prompt(token_ids, mm_data, mm_uuids)
