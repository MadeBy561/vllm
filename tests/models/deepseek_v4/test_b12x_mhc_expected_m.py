# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.models.deepseek_v4.nvidia.model as deepseek_v4_model
from vllm.config import CUDAGraphMode
from vllm.models.deepseek_v4.nvidia.model import (
    DeepseekV4DecoderLayer,
    _b12x_mhc_expected_m,
)


def test_b12x_mhc_expected_m_uses_live_decode_m() -> None:
    capture_sizes = [1, 2, 4, 8, 16, 32, 64, 80, 88, 96]

    assert (
        _b12x_mhc_expected_m(
            80,
            is_prefill=False,
            max_num_batched_tokens=4096,
            cudagraph_capture_sizes=capture_sizes,
        )
        == 80
    )
    assert (
        _b12x_mhc_expected_m(
            90,
            is_prefill=False,
            max_num_batched_tokens=4096,
            cudagraph_capture_sizes=capture_sizes,
        )
        == 90
    )


def test_b12x_mhc_expected_m_keeps_small_prefill_live_sized() -> None:
    capture_sizes = [1, 2, 4, 8, 16, 32, 64, 80, 88, 96]

    assert (
        _b12x_mhc_expected_m(
            3,
            is_prefill=True,
            max_num_batched_tokens=4096,
            cudagraph_capture_sizes=capture_sizes,
        )
        == 3
    )
    assert (
        _b12x_mhc_expected_m(
            80,
            is_prefill=True,
            max_num_batched_tokens=4096,
            cudagraph_capture_sizes=capture_sizes,
        )
        == 80
    )
    assert (
        _b12x_mhc_expected_m(
            96,
            is_prefill=True,
            max_num_batched_tokens=4096,
            cudagraph_capture_sizes=capture_sizes,
        )
        == 4096
    )


def test_b12x_mhc_expected_m_falls_back_to_prefill_crossover() -> None:
    capture_sizes = [1, 2, 4, 8, 16, 32, 64, 80, 88, 96]

    assert (
        _b12x_mhc_expected_m(
            90,
            is_prefill=None,
            max_num_batched_tokens=4096,
            cudagraph_capture_sizes=capture_sizes,
        )
        == 90
    )
    assert (
        _b12x_mhc_expected_m(
            96,
            is_prefill=None,
            max_num_batched_tokens=4096,
            cudagraph_capture_sizes=capture_sizes,
        )
        == 4096
    )


def test_b12x_mhc_expected_m_full_decode_graph_uses_live_m(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = object.__new__(DeepseekV4DecoderLayer)
    forward_context = SimpleNamespace(
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        batch_descriptor=SimpleNamespace(uniform=True),
    )

    monkeypatch.setattr(deepseek_v4_model, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(
        deepseek_v4_model, "get_forward_context", lambda: forward_context
    )
    monkeypatch.setattr(
        DeepseekV4DecoderLayer,
        "_b12x_mhc_prefill_state",
        lambda _: pytest.fail("full decode graph should not query prefill metadata"),
    )

    assert layer._b12x_mhc_expected_m(1) == 1


def test_b12x_mhc_requires_fused_norm_weight() -> None:
    layer = object.__new__(DeepseekV4DecoderLayer)

    with pytest.raises(RuntimeError, match="requires fused RMSNorm"):
        layer._require_b12x_mhc_norm_weight(None)

    norm_weight = torch.ones(4)

    assert layer._require_b12x_mhc_norm_weight(norm_weight) is norm_weight
