# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.deepseek_v4.nvidia.model import (
    DeepseekV4DecoderLayer,
    _b12x_mhc_expected_m,
)


def test_b12x_mhc_expected_m_uses_decode_capture_bucket() -> None:
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
        == 96
    )


def test_b12x_mhc_expected_m_uses_prefill_chunk_for_small_tail() -> None:
    capture_sizes = [1, 2, 4, 8, 16, 32, 64, 80, 88, 96]

    assert (
        _b12x_mhc_expected_m(
            80,
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
        == 96
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


def test_b12x_mhc_requires_fused_norm_weight() -> None:
    layer = object.__new__(DeepseekV4DecoderLayer)

    with pytest.raises(RuntimeError, match="requires fused RMSNorm"):
        layer._require_b12x_mhc_norm_weight(None)

    norm_weight = torch.ones(4)

    assert layer._require_b12x_mhc_norm_weight(norm_weight) is norm_weight
