# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import types

import pytest
import torch

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.model_executor.kernels.linear import (
    _LINEAR_BACKEND_KERNEL_MAP,
    _POSSIBLE_MXFP8_KERNELS,
    init_mxfp8_linear_kernel,
)
from vllm.model_executor.kernels.linear.mxfp8.b12x import (
    B12xMxfp8LinearKernel,
    _b12x_mxfp8_linear,
)
from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
    Mxfp8LinearLayerConfig,
)
from vllm.platforms import PlatformEnum


class _Param:
    def __init__(self, data: torch.Tensor) -> None:
        self.data = data


def test_b12x_backend_maps_mxfp8_kernel() -> None:
    assert B12xMxfp8LinearKernel in _LINEAR_BACKEND_KERNEL_MAP["b12x"]
    assert B12xMxfp8LinearKernel in _POSSIBLE_MXFP8_KERNELS[PlatformEnum.CUDA]


def test_b12x_mxfp8_explicit_backend_selects_kernel(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear as linear_mod

    monkeypatch.setattr(linear_mod.current_platform, "_enum", PlatformEnum.CUDA)
    monkeypatch.setattr(linear_mod, "_get_linear_backend", lambda: "b12x")
    monkeypatch.setattr(
        B12xMxfp8LinearKernel,
        "is_supported",
        classmethod(lambda cls, compute_capability=None: (True, None)),
    )
    monkeypatch.setattr(
        B12xMxfp8LinearKernel,
        "can_implement",
        classmethod(lambda cls, c: (True, None)),
    )

    kernel = init_mxfp8_linear_kernel()

    assert isinstance(kernel, B12xMxfp8LinearKernel)


def test_b12x_mxfp8_env_selects_kernel_with_auto_backend(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear as linear_mod

    monkeypatch.setattr(linear_mod.current_platform, "_enum", PlatformEnum.CUDA)
    monkeypatch.setattr(linear_mod, "_get_linear_backend", lambda: "auto")
    monkeypatch.setattr(linear_mod.envs, "VLLM_USE_B12X_FP8_GEMM", True)
    monkeypatch.setattr(
        B12xMxfp8LinearKernel,
        "is_supported",
        classmethod(lambda cls, compute_capability=None: (True, None)),
    )
    monkeypatch.setattr(
        B12xMxfp8LinearKernel,
        "can_implement",
        classmethod(lambda cls, c: (True, None)),
    )

    kernel = init_mxfp8_linear_kernel()

    assert isinstance(kernel, B12xMxfp8LinearKernel)


def test_b12x_mxfp8_env_rejects_incompatible_backend(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear as linear_mod

    monkeypatch.setattr(linear_mod.current_platform, "_enum", PlatformEnum.CUDA)
    monkeypatch.setattr(linear_mod, "_get_linear_backend", lambda: "marlin")
    monkeypatch.setattr(linear_mod.envs, "VLLM_USE_B12X_FP8_GEMM", True)

    with pytest.raises(ValueError, match="requires --linear-backend=auto or b12x"):
        init_mxfp8_linear_kernel()


def test_b12x_mxfp8_can_implement_requires_opt_in(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.mxfp8.b12x as b12x_mod

    monkeypatch.setattr(b12x_mod, "_current_linear_backend", lambda: "auto")
    monkeypatch.setattr(b12x_mod.envs, "VLLM_USE_B12X_FP8_GEMM", False)

    can_implement, reason = B12xMxfp8LinearKernel.can_implement(
        Mxfp8LinearLayerConfig()
    )

    assert not can_implement
    assert reason == "b12x MXFP8 GEMM is not enabled"

    monkeypatch.setattr(b12x_mod, "_current_linear_backend", lambda: "b12x")
    can_implement, reason = B12xMxfp8LinearKernel.can_implement(
        Mxfp8LinearLayerConfig()
    )

    assert can_implement
    assert reason is None


def test_b12x_mxfp8_disabled_support_check_skips_import(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.mxfp8.b12x as b12x_mod

    monkeypatch.setattr(b12x_mod.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        b12x_mod.current_platform,
        "is_device_capability_family",
        lambda family: family == 120,
    )
    monkeypatch.setattr(b12x_mod, "_current_linear_backend", lambda: "auto")
    monkeypatch.setattr(b12x_mod.envs, "VLLM_USE_B12X_FP8_GEMM", False)

    def fail_import():
        raise AssertionError("B12X MXFP8 import should require opt-in")

    monkeypatch.setattr(b12x_mod, "_import_b12x_mxfp8", fail_import)

    is_supported, reason = B12xMxfp8LinearKernel.is_supported()

    assert not is_supported
    assert reason == "b12x MXFP8 GEMM is not enabled"


def test_b12x_mxfp8_support_requires_runtime_api(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.mxfp8.b12x as b12x_mod

    monkeypatch.setattr(b12x_mod.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        b12x_mod.current_platform,
        "is_device_capability_family",
        lambda family: family == 120,
    )
    monkeypatch.setattr(b12x_mod, "_current_linear_backend", lambda: "b12x")
    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_mxfp8",
        lambda: types.SimpleNamespace(pack_mxfp8_linear_weight=object()),
    )

    is_supported, reason = B12xMxfp8LinearKernel.is_supported()

    assert not is_supported
    assert reason == "b12x.gemm.mxfp8_linear missing callable pack_mxfp8_linear_weight"


def test_b12x_mxfp8_process_weights_packs_modelopt_layout(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.mxfp8.b12x as b12x_mod

    calls = []
    packed = types.SimpleNamespace(out_features=48)

    def pack(weight: torch.Tensor, weight_scale: torch.Tensor):
        calls.append((weight, weight_scale))
        return packed

    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_mxfp8",
        lambda: types.SimpleNamespace(pack_mxfp8_linear_weight=pack),
    )

    layer = torch.nn.Module()
    layer.prefix = "model.layers.0.self_attn.qkv_proj"
    layer.weight = _Param(torch.empty((48, 128), dtype=torch.float8_e4m3fn))
    layer.weight_scale = _Param(torch.empty((64, 8), dtype=torch.uint8))
    kernel = object.__new__(B12xMxfp8LinearKernel)
    vllm_config = VllmConfig()

    monkeypatch.setattr(
        b12x_mod, "get_current_vllm_config_or_none", lambda: vllm_config
    )

    kernel.process_weights_after_loading(layer)

    assert layer.b12x_mxfp8_packed_weight is packed
    assert (
        vllm_config.compilation_config.static_forward_context[layer.prefix] is layer
    )
    assert len(calls) == 1
    weight, weight_scale = calls[0]
    assert weight.shape == (48, 128)
    assert weight_scale.shape == (48, 4)
    assert weight.dtype == torch.float8_e4m3fn
    assert weight_scale.dtype == torch.uint8


def test_b12x_mxfp8_apply_uses_packed_weight(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.mxfp8.b12x as b12x_mod

    calls = []

    def mxfp8_linear(
        source: torch.Tensor,
        packed_weight,
        *,
        bias: torch.Tensor | None = None,
        expected_m: int | None = None,
    ) -> torch.Tensor:
        calls.append((source, packed_weight, bias, expected_m))
        return source.new_full((source.shape[0], packed_weight.out_features), 3.0)

    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_mxfp8",
        lambda: types.SimpleNamespace(mxfp8_linear=mxfp8_linear),
    )

    layer = torch.nn.Module()
    packed = types.SimpleNamespace(out_features=48)
    layer.b12x_mxfp8_packed_weight = packed
    x = torch.empty((2, 3, 128), dtype=torch.bfloat16)
    bias = torch.empty((48,), dtype=torch.bfloat16)
    kernel = object.__new__(B12xMxfp8LinearKernel)

    output = kernel.apply_weights(layer, x, bias)

    assert output.shape == (2, 3, 48)
    assert output.dtype == x.dtype
    assert len(calls) == 1
    source, called_packed, called_bias, expected_m = calls[0]
    assert source.shape == (6, 128)
    assert called_packed is packed
    assert called_bias is bias
    assert expected_m == 6


def test_b12x_mxfp8_compile_path_uses_forward_context_custom_op(
    monkeypatch,
) -> None:
    calls = []

    def op(
        x: torch.Tensor,
        bias: torch.Tensor | None,
        layer_name: str,
        out_features: int,
    ) -> torch.Tensor:
        calls.append((x, bias, layer_name, out_features))
        return x.new_full((*x.shape[:-1], out_features), 7.0)

    monkeypatch.setattr(torch.ops.vllm, "b12x_mxfp8_linear", op)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    layer = torch.nn.Module()
    layer.prefix = "model.layers.1.mlp.gate_up_proj"
    packed = types.SimpleNamespace(out_features=32)
    layer.b12x_mxfp8_packed_weight = packed
    x = torch.empty((2, 3, 128), dtype=torch.bfloat16)
    bias = torch.empty((32,), dtype=torch.bfloat16)
    kernel = object.__new__(B12xMxfp8LinearKernel)

    output = kernel.apply_weights(layer, x, bias)

    assert output.shape == (2, 3, 32)
    assert output.dtype == x.dtype
    assert len(calls) == 1
    source, called_bias, layer_name, out_features = calls[0]
    assert source is x
    assert called_bias is bias
    assert getattr(layer_name, "value", layer_name) == layer.prefix
    assert out_features == 32
    torch.testing.assert_close(output, torch.full_like(output, 7.0))


def test_b12x_mxfp8_custom_op_body_uses_forward_context(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.mxfp8.b12x as b12x_mod

    calls = []

    def mxfp8_linear(
        source: torch.Tensor,
        packed_weight,
        *,
        bias: torch.Tensor | None = None,
        expected_m: int | None = None,
    ) -> torch.Tensor:
        calls.append((source, packed_weight, bias, expected_m))
        return source.new_full((source.shape[0], packed_weight.out_features), 11.0)

    monkeypatch.setattr(
        b12x_mod,
        "_import_b12x_mxfp8",
        lambda: types.SimpleNamespace(mxfp8_linear=mxfp8_linear),
    )

    layer = torch.nn.Module()
    layer.prefix = "model.layers.2.mlp.down_proj"
    packed = types.SimpleNamespace(out_features=16)
    layer.b12x_mxfp8_packed_weight = packed
    x = torch.empty((2, 3, 128), dtype=torch.bfloat16)
    bias = torch.empty((16,), dtype=torch.bfloat16)
    vllm_config = VllmConfig()
    vllm_config.compilation_config.static_forward_context[layer.prefix] = layer

    with set_forward_context({}, vllm_config):
        output = _b12x_mxfp8_linear(x, bias, layer.prefix, 16)

    assert output.shape == (2, 3, 16)
    assert len(calls) == 1
    source, called_packed, called_bias, expected_m = calls[0]
    assert source.shape == (6, 128)
    assert called_packed is packed
    assert called_bias is bias
    assert expected_m == 6
    torch.testing.assert_close(output, torch.full_like(output, 11.0))
