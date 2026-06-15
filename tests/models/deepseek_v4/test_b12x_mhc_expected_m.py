# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.deepseek_v4.nvidia.model import DeepseekV4DecoderLayer


def _make_b12x_layer() -> DeepseekV4DecoderLayer:
    return object.__new__(DeepseekV4DecoderLayer)


def test_b12x_mhc_requires_fused_norm_weight() -> None:
    layer = _make_b12x_layer()

    with pytest.raises(RuntimeError, match="requires fused RMSNorm"):
        layer._require_b12x_mhc_norm_weight(None)

    norm_weight = torch.ones(4)

    assert layer._require_b12x_mhc_norm_weight(norm_weight) is norm_weight
