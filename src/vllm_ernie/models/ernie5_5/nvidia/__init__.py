# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ERNIE 5.5 hybrid HCA/DSA model (``Ernie5_5_MoeForCausalLM``)."""

from vllm.platforms import current_platform

if current_platform.is_rocm() or current_platform.is_xpu():
    raise NotImplementedError("ERNIE 5.5 only supports NVIDIA GPUs.")

from .dspark import Ernie5_5DSparkForCausalLM  # noqa: E402
from .model import Ernie5_5ForCausalLM  # noqa: E402
from .mtp import Ernie5_5MTP  # noqa: E402

__all__ = [
    "Ernie5_5DSparkForCausalLM",
    "Ernie5_5ForCausalLM",
    "Ernie5_5MTP",
]
