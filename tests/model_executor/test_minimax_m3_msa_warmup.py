# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import vllm.model_executor.warmup.minimax_m3_msa_warmup as msa_warmup


def test_minimax_m3_triton_msa_warmup_supports_blackwell(monkeypatch) -> None:
    platform = SimpleNamespace(
        is_cuda=lambda: True,
        is_device_capability_family=lambda family: family == 120,
    )

    monkeypatch.setattr(msa_warmup, "current_platform", platform)
    monkeypatch.setattr(msa_warmup.envs, "VLLM_USE_B12X_MINIMAX_M3_MSA", False)

    assert msa_warmup._supports_minimax_m3_msa_warmup()


def test_minimax_m3_b12x_msa_warmup_stays_blackwell_only(monkeypatch) -> None:
    platform = SimpleNamespace(
        is_cuda=lambda: True,
        is_device_capability_family=lambda family: family == 100,
    )

    monkeypatch.setattr(msa_warmup, "current_platform", platform)
    monkeypatch.setattr(msa_warmup.envs, "VLLM_USE_B12X_MINIMAX_M3_MSA", True)

    assert not msa_warmup._supports_minimax_m3_msa_warmup()
