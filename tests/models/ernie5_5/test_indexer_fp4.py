# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config import AttentionConfig, CacheConfig, ModelConfig, VllmConfig
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import mxfp4_quantize
from vllm.platforms import current_platform

from vllm_ernie.models.ernie5_5.config import Ernie5_5Config
from vllm_ernie.models.ernie5_5.nvidia.attention import Ernie5_5Indexer

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.is_device_capability_family(100)),
    reason="The ERNIE MXFP4 indexer requires SM100-family GPUs",
)
PREFIX = "model.layers.0.self_attn.indexer.k_cache"


@pytest.fixture(scope="module")
def model_config(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("ernie_fp4")
    Ernie5_5Config(
        hidden_size=128, hybrid_mla_q_lora_rank=128, num_hidden_layers=1,
        num_nextn_predict_layers=0, compress_ratios=[-2],
        max_position_embeddings=4096, architectures=["Ernie5_5_MoeForCausalLM"],
    ).save_pretrained(model_dir)
    return ModelConfig(model=str(model_dir), skip_tokenizer_init=True,
                       dtype="bfloat16", max_model_len=4096)


def _config(model_config, pcp=1, dcp=1):
    config = VllmConfig(model_config=model_config,
                        cache_config=CacheConfig(block_size=64),
                        attention_config=AttentionConfig(indexer_kv_dtype="mxfp4"))
    config.parallel_config.prefill_context_parallel_size = pcp
    config.parallel_config.decode_context_parallel_size = dcp
    return config


@pytest.mark.parametrize("pcp,dcp", [(2, 1), (1, 2), (2, 2)])
def test_mxfp4_requires_single_context_parallel(model_config, pcp, dcp):
    with pytest.raises(NotImplementedError, match="PCP=DCP=1") as error:
        Ernie5_5Indexer(_config(model_config, pcp, dcp), model_config.hf_config,
                        None, CacheConfig(block_size=64), None, None)
    assert f"PCP={pcp}, DCP={dcp}" in str(error.value)


def test_mxfp4_quantizes_q_and_writes_k_cache():
    q = torch.randn(2, 64, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
    storage = torch.full((1, 9216), 173, device="cuda", dtype=torch.uint8)
    cache = storage.as_strided((1, 64, 68), (9216, 68, 1))
    slots = torch.tensor([0, 1], device="cuda")
    indexer = SimpleNamespace(
        n_head=64, head_dim=128,
        k_cache=SimpleNamespace(prefix=PREFIX, kv_cache=cache),
    )
    context = ForwardContext({}, {PREFIX: SimpleNamespace(slot_mapping=slots)}, {})
    with override_forward_context(context):
        values, scales = Ernie5_5Indexer._quantize_fp4(indexer, q.flatten(0, 1), k)
    expected_q, expected_q_scales = mxfp4_quantize(q)
    expected_k, expected_k_scales = mxfp4_quantize(k)
    torch.testing.assert_close(values, expected_q)
    torch.testing.assert_close(scales, expected_q_scales.view(torch.int32).squeeze(-1))
    torch.testing.assert_close(storage[0, :128].view(2, 64), expected_k)
    torch.testing.assert_close(storage[0, 4096:4104].view(2, 4), expected_k_scales)
