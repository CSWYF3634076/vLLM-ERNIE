# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the hooks that patch shared vLLM code for ERNIE 5.5."""

import asyncio
import pickle
from types import SimpleNamespace
from typing import get_args

import pytest

import vllm.config.speculative as spec_mod
from vllm.config import SpeculativeConfig
from vllm.config.vllm import default_breakable_cudagraph_architectures
from vllm.model_executor.models.registry import ModelRegistry
from vllm.transformers_utils.config import _CONFIG_REGISTRY
from vllm.transformers_utils.model_arch_config_convertor import (
    MODEL_ARCH_CONFIG_CONVERTORS,
)

from vllm_ernie.models import ernie5_5
from vllm_ernie.models.ernie5_5.config import Ernie5_5Config
from vllm_ernie.models.ernie5_5.patches import _undo_dspark_rewrite


def test_ernie5_5_dsa_layers_use_hybrid_mla_dims():
    """DSA layers (``compress_ratios <= 0``) must not read the HCA dims."""
    config = Ernie5_5Config(
        num_hidden_layers=3,
        compress_ratios=[128, -2, 128, -2],
        hybrid_mla_qk_nope_head_dim=192,
        hybrid_mla_v_head_dim=256,
    )

    assert [config.is_dsa_layer(i) for i in range(4)] == [False, True, False, True]
    assert (config.mla_qk_nope_head_dim, config.qk_nope_head_dim) == (192, 448)
    assert config.mla_v_head_dim == 256
    assert config.rope_parameters["rope_type"] == "default"
    assert config.head_dtype == "float32"


def test_architectures_and_config_registered():
    supported = ModelRegistry.get_supported_archs()
    assert set(ernie5_5.ARCHITECTURES) <= set(supported)
    assert _CONFIG_REGISTRY["ernie5_5_moe"] is Ernie5_5Config
    breakable = default_breakable_cudagraph_architectures()
    assert {"Ernie5_5_MoeForCausalLM", "Ernie5_5MoeMTPModel"} <= breakable


def test_is_deepseek_mla():
    config = Ernie5_5Config(num_hidden_layers=2, compress_ratios=[128, -2])
    for model_type in ("ernie5_5_moe", "ernie5_5_moe_mtp"):
        convertor = MODEL_ARCH_CONFIG_CONVERTORS[model_type](config, config)
        assert convertor.is_deepseek_mla()


def test_mtp_hf_config_override():
    assert "ernie5_5_moe_mtp" in get_args(spec_mod.MTPModelTypes)
    config = Ernie5_5Config(num_hidden_layers=2, num_nextn_predict_layers=1)
    config.architectures = ["Ernie5_5_MoeForCausalLM"]
    config = SpeculativeConfig.hf_config_override(config)
    assert config.model_type == "ernie5_5_moe_mtp"
    assert config.architectures == ["Ernie5_5MoeMTPModel"]
    assert config.n_predict == 1

    # The draft ModelConfig carries this function to engine-core processes.
    fn = pickle.loads(pickle.dumps(SpeculativeConfig.hf_config_override))
    assert fn is SpeculativeConfig.hf_config_override


def test_dspark_draft_keeps_own_config():
    config = Ernie5_5Config(num_hidden_layers=2)
    config.architectures = ["Ernie5_5_MoeDSparkDraftModel"]
    assert SpeculativeConfig.hf_config_override(config).model_type == "ernie5_5_moe"

    # State after SpeculativeConfig rewrites the DSpark draft.
    verified = []
    draft = SimpleNamespace(
        architectures=["Ernie5_5_MoeDSparkDraftModel"],
        hf_config=SimpleNamespace(
            model_type="deepseek_v4", architectures=["DSparkDraftModel"]
        ),
        quantization="fp8",
        _verify_quantization=lambda: verified.append(True),
    )
    spec = SimpleNamespace(method="dspark", draft_model_config=draft, quantization=None)
    _undo_dspark_rewrite(spec)
    assert draft.hf_config.model_type == "ernie5_5_moe"
    assert draft.hf_config.architectures == ["Ernie5_5_MoeDSparkDraftModel"]
    assert draft.quantization is None
    assert verified


def test_bundled_dspark_draft_reshapes_target_config():
    config = Ernie5_5Config(
        num_hidden_layers=4,
        num_nextn_predict_layers=0,
        compress_ratios=[128, -2, 128, 4],
        dspark_target_layer_ids=[1, 2, 3],
        dspark_noise_token_id=1024,
        dspark_markov_rank=256,
    )
    config.architectures = ["DSparkDraftModel"]
    draft = SimpleNamespace(
        architectures=["DSparkDraftModel"],
        hf_config=config,
        quantization="fp8",
        _verify_quantization=lambda: None,
    )
    spec = SimpleNamespace(method="dspark", draft_model_config=draft, quantization=None)
    _undo_dspark_rewrite(spec)
    assert config.architectures == ["Ernie5_5_MoeDSparkDraftModel"]
    assert config.model_type == "ernie5_5_moe"
    assert config.num_hidden_layers == 3
    assert config.compress_ratios == [128] * 3
    assert config.first_k_dense_replace == 0
    assert config.mask_token_id == 1024
    assert config.enable_confidence_head
    assert config.dspark_bundled_in_target
    assert not config.bundles_dspark_draft


def test_unbundle_dspark_weights():
    from vllm_ernie.models.ernie5_5.nvidia.dspark import _unbundle_dspark_weights

    names = [
        "model.layers.0.input_layernorm.weight",
        "model.norm.weight",
        "lm_head.weight",
        "model.mtp.2.input_layernorm.weight",
        "model.mtp.markov_head.markov_w1.weight",
        "model.mtp.mhc_contract.hc_head_fn",
    ]
    out = [name for name, _ in _unbundle_dspark_weights((n, None) for n in names)]
    assert out == [
        "model.layers.2.input_layernorm.weight",
        "model.markov_head.markov_w1.weight",
        "model.mhc_contract.hc_head_fn",
    ]


class _FakeRequest:
    def __init__(self, status, params):
        self.request_id = "r"
        self.status = status
        self.kv_transfer_params = params


def _mooncake_scheduler(model_type: str):
    from vllm.distributed.kv_transfer.kv_connector.factory import (
        KVConnectorFactory,
    )

    connector_cls = KVConnectorFactory.get_connector_class_by_name("MooncakeConnector")
    from vllm.distributed.kv_transfer.kv_connector.v1.mooncake import (
        mooncake_connector,
    )

    assert connector_cls is mooncake_connector.MooncakeConnector
    scheduler_cls = mooncake_connector.MooncakeConnectorScheduler
    scheduler = scheduler_cls.__new__(scheduler_cls)
    scheduler.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type=model_type))
    )
    scheduler.is_kv_consumer = False
    scheduler._reqs_need_send = {}
    scheduler._reqs_not_processed = set()
    scheduler.get_sw_clipped_blocks = lambda block_ids: block_ids
    return scheduler


@pytest.mark.parametrize(
    ("model_type", "status_name", "sends"),
    [
        ("ernie5_5_moe", "FINISHED_LENGTH_CAPPED", True),
        # The single prefill token was EOS / a stop token.
        ("ernie5_5_moe", "FINISHED_STOPPED", True),
        ("ernie5_5_moe", "FINISHED_ABORTED", False),
        # Other models keep vLLM's behavior.
        ("deepseek_v4", "FINISHED_STOPPED", False),
    ],
)
def test_mooncake_sends_stopped_prefill(model_type, status_name, sends):
    from vllm.v1.request import RequestStatus

    scheduler = _mooncake_scheduler(model_type)
    status = RequestStatus[status_name]
    request = _FakeRequest(status, {"do_remote_decode": True, "transfer_id": "t"})

    delay_free, _ = scheduler.request_finished(request, ([3, 4],))

    assert delay_free is sends
    assert ("r" in scheduler._reqs_need_send) is sends
    assert ("t" in scheduler._reqs_not_processed) is not sends
    assert request.status is status


class _StopRender(Exception):
    pass


class _FakeOnlineRenderer:
    def __init__(self, model_type: str):
        self.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(model_type=model_type)
        )

    @property
    def renderer(self):
        # Stop right after the ERNIE hook, before any real rendering.
        raise _StopRender


_TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {}}}]


@pytest.mark.parametrize(
    ("model_type", "parallel", "tools", "user_options", "expected"),
    [
        ("ernie5_5_moe", False, _TOOLS, None, {"parallel_tool_calls": False}),
        ("ernie5_5_moe", True, _TOOLS, None, None),
        ("ernie5_5_moe", False, None, None, None),
        # An explicit option from the user wins.
        (
            "ernie5_5_moe",
            False,
            _TOOLS,
            {"parallel_tool_calls": True, "thinking_mode": "close"},
            {"parallel_tool_calls": True, "thinking_mode": "close"},
        ),
        ("deepseek_v4", False, _TOOLS, None, None),
    ],
)
def test_parallel_tool_calls_false_reaches_template(
    model_type, parallel, tools, user_options, expected
):
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.renderers.online_renderer import OnlineRenderer

    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        model="m",
        tools=tools,
        parallel_tool_calls=parallel,
        chat_template_kwargs=(
            {"options": user_options} if user_options is not None else None
        ),
    )
    with pytest.raises(_StopRender):
        asyncio.run(
            OnlineRenderer.preprocess_chat(
                _FakeOnlineRenderer(model_type), request, [], None, "auto", None
            )
        )

    kwargs = request.build_chat_params(None, "auto").chat_template_kwargs
    assert kwargs.get("options") == expected
