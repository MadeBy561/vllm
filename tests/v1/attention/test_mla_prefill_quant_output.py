# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MLA prefill backend fused-quant-output support.

Covers two things:
  * `MLAPrefillBackend.supports_quant_output`, the capability gate that decides
    whether the prefill kernel writes quantized output directly (FA4 native
    fused FP8, see flash-attention#135) instead of the post-quant path.
  * The numerical equivalence of that fused FP8 write versus the bf16-attention
    + standalone static-FP8-quant path it replaces (GPU-only, SM100/SM110).
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Dynamic128Sym,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend
from vllm.v1.attention.backends.mla.prefill.flash_attn import (
    FlashAttnPrefillBackend,
)

_FA_MODULE = "vllm.v1.attention.backends.mla.prefill.flash_attn"


class _DummyPrefillBackend(MLAPrefillBackend):
    """Concrete backend that does NOT override supports_quant_output."""

    @staticmethod
    def get_name() -> str:
        return "DUMMY"

    def run_prefill_new_tokens(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def run_prefill_context_chunk(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


@pytest.mark.parametrize(
    "quant_key", [kFp8StaticTensorSym, kFp8Dynamic128Sym, kNvfp4Dynamic, None]
)
def test_base_backend_never_supports_quant_output(quant_key):
    """The base default opts every backend out unless it overrides."""
    backend = object.__new__(_DummyPrefillBackend)
    assert backend.supports_quant_output(quant_key) is False


def _make_fa_backend(version: int | None, is_vllm_fa: bool):
    """Build a FlashAttnPrefillBackend without running its heavy __init__."""
    backend = object.__new__(FlashAttnPrefillBackend)
    backend.vllm_flash_attn_version = version
    backend._is_vllm_fa = is_vllm_fa
    return backend


@pytest.mark.parametrize(
    ("version", "is_vllm_fa", "dc_major", "quant_key", "expected"),
    [
        # FA4 + vLLM-FA + Blackwell SM100/SM110 + static FP8 -> fused.
        (4, True, 10, kFp8StaticTensorSym, True),
        (4, True, 11, kFp8StaticTensorSym, True),
        # Wrong compute capability (SM90 / SM120) -> not supported (#135).
        (4, True, 9, kFp8StaticTensorSym, False),
        (4, True, 12, kFp8StaticTensorSym, False),
        # Not FA4.
        (3, True, 10, kFp8StaticTensorSym, False),
        (2, True, 10, kFp8StaticTensorSym, False),
        (None, True, 10, kFp8StaticTensorSym, False),
        # Upstream (ROCm) flash-attn, not vLLM-FA.
        (4, False, 10, kFp8StaticTensorSym, False),
        # Quant keys not wired through FA4 yet.
        (4, True, 10, kFp8Dynamic128Sym, False),
        (4, True, 10, kNvfp4Dynamic, False),
    ],
)
def test_flash_attn_supports_quant_output(
    version, is_vllm_fa, dc_major, quant_key, expected
):
    backend = _make_fa_backend(version, is_vllm_fa)
    with patch(f"{_FA_MODULE}.current_platform") as plat:
        plat.get_device_capability.return_value = DeviceCapability(
            major=dc_major, minor=0
        )
        assert backend.supports_quant_output(quant_key) is expected


def test_flash_attn_supports_quant_output_unknown_device():
    """A None device capability (e.g. capability probe failed) is safe."""
    backend = _make_fa_backend(version=4, is_vllm_fa=True)
    with patch(f"{_FA_MODULE}.current_platform") as plat:
        plat.get_device_capability.return_value = None
        assert backend.supports_quant_output(kFp8StaticTensorSym) is False


@pytest.mark.parametrize(
    ("version", "enable_jit_warmup", "expected_calls"),
    [(4, True, 1), (4, False, 0), (3, True, 0)],
)
def test_flash_attn_registers_warmup_only_for_fa4(
    version: int,
    enable_jit_warmup: bool,
    expected_calls: int,
):
    vllm_config = SimpleNamespace(
        kernel_config=SimpleNamespace(enable_jit_warmup=enable_jit_warmup)
    )
    with (
        patch(f"{_FA_MODULE}.flash_attn_varlen_func"),
        patch(f"{_FA_MODULE}.get_flash_attn_version", return_value=version),
        patch(f"{_FA_MODULE}._FA4_MLA_PREFILL_KERNEL.register_warmup") as register,
        patch(f"{_FA_MODULE}.current_platform") as platform,
    ):
        platform.get_device_capability.return_value = DeviceCapability(10, 0)
        FlashAttnPrefillBackend(
            num_heads=16,
            scale=1.0,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            vllm_config=vllm_config,
        )

    assert register.call_count == expected_calls


def test_flash_attn_prefill_backend_signature_accepts_fused_kwargs():
    """run_prefill_new_tokens must accept out/output_scale so the direct
    (non-**kwargs) call in forward_mha type- and runtime-checks."""
    import inspect

    params = inspect.signature(
        FlashAttnPrefillBackend.run_prefill_new_tokens
    ).parameters
    assert "out" in params
    assert "output_scale" in params
    # The base contract must expose them too (Liskov / direct call site).
    base_params = inspect.signature(MLAPrefillBackend.run_prefill_new_tokens).parameters
    assert "out" in base_params
    assert "output_scale" in base_params


def test_mla_impl_forward_mha_accepts_output_scale():
    """The abstract MLA impl forward_mha must carry output_scale so every
    override (and the unconditional forward_impl call) stays compatible."""
    import inspect

    from vllm.v1.attention.backend import MLAAttentionImpl

    params = inspect.signature(MLAAttentionImpl.forward_mha).parameters
    assert "output_scale" in params
    assert params["output_scale"].default is None


def _fused_fp8_skip_reason() -> str | None:
    """FA4 fused FP8 output needs a real Blackwell SM100/SM110 GPU."""
    if not torch.cuda.is_available():
        return "requires CUDA"
    major = torch.cuda.get_device_capability()[0]
    if major not in (10, 11):
        return f"FA4 fused FP8 output requires SM100/SM110, got SM{major}x"
    return None


_FUSED_FP8_SKIP = _fused_fp8_skip_reason()


@pytest.mark.skipif(_FUSED_FP8_SKIP is not None, reason=_FUSED_FP8_SKIP or "")
def test_fa4_fused_fp8_output_matches_post_quant(default_vllm_config):
    """FA4's fused FP8 write (output_scale, flash-attention#135) must match the
    bf16-attention + standalone static-FP8-quant path it replaces, since
    production uses the same output_scale for both."""
    from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
    from vllm.platforms import current_platform
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    torch.manual_seed(0)
    device = torch.device("cuda")
    fp8_dtype = current_platform.fp8_dtype()

    # MLA prefill head dims (post kv_b_proj): q/k = qk_nope(128)+qk_rope(64),
    # v = v_head_dim(128); DeepSeek-V2-Lite has 16 query heads.
    num_heads, qk_head_dim, v_head_dim, seqlen = 16, 192, 128, 512
    cu_seqlens = torch.tensor([0, seqlen], dtype=torch.int32, device=device)
    q = torch.randn(seqlen, num_heads, qk_head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(seqlen, num_heads, qk_head_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(seqlen, num_heads, v_head_dim, dtype=torch.bfloat16, device=device)

    fa_kwargs = dict(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=seqlen,
        max_seqlen_k=seqlen,
        causal=True,
        fa_version=4,
    )

    # Reference: bf16 attention, then standalone static per-tensor FP8 quant.
    out_bf16 = flash_attn_varlen_func(q=q, k=k, v=v, **fa_kwargs)
    out_2d = out_bf16.reshape(seqlen, num_heads * v_head_dim)
    # Scale the amax near e4m3 max so the check uses the representable range.
    finfo = torch.finfo(fp8_dtype)
    scale = (out_2d.abs().max() / finfo.max).to(torch.float32).reshape(1)
    quant_op = QuantFP8(static=True, group_shape=GroupShape.PER_TENSOR)
    ref_fp8, _ = quant_op(out_2d, scale)

    # Feature: FA4 writes e4m3 into the (tokens, heads*dim) buffer directly.
    fused_fp8 = torch.empty(
        seqlen, num_heads * v_head_dim, dtype=fp8_dtype, device=device
    )
    flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        out=fused_fp8.view(seqlen, num_heads, v_head_dim),
        output_scale=scale,
        **fa_kwargs,
    )

    # Non-degenerate (catches a no-op / all-zero write).
    assert torch.isfinite(fused_fp8.float()).all()
    assert fused_fp8.float().abs().any()

    # e4m3 has 3 mantissa bits, so allow ~1 mantissa step of rounding slack.
    ref = ref_fp8.float() * scale
    got = fused_fp8.float() * scale
    torch.testing.assert_close(got, ref, rtol=0.125, atol=float(scale) * 2)

    # ...and most elements land in the exact same fp8 bucket.
    exact = (fused_fp8.view(torch.uint8) == ref_fp8.view(torch.uint8)).float().mean()
    assert exact > 0.9, f"only {exact:.1%} of fused FP8 outputs matched the baseline"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("causal", [True, False])
@torch.inference_mode()
def test_b12x_prefill_packed_sequences_and_changed_input_graph(causal, monkeypatch):
    """Prepared prefill preserves packed boundaries and mutable graph inputs."""
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("requires SM12x")
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.preparation import PreparationSession

    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonMetadataBuilder,
    )
    from vllm.v1.attention.backends.mla.prefill.b12x import B12xPrefillBackend
    from vllm.v1.worker import workspace

    device = torch.device("cuda", torch.accelerator.current_device_index())
    manager = workspace.WorkspaceManager(device)
    monkeypatch.setattr(workspace, "_manager", manager)
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    monkeypatch.setattr(
        MLACommonMetadataBuilder,
        "determine_chunked_prefill_workspace_size",
        staticmethod(lambda _: 128),
    )
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32, max_num_seqs=2),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
    )
    heads, scale = 10, 192**-0.5
    backend = B12xPrefillBackend(heads, scale, 512, 128, 64, 128, config)
    units = backend.get_b12x_preparation_units(
        SimpleNamespace(layer_name="test.mla"), SimpleNamespace(stage="weights")
    )
    query_lengths = [3, 6]
    key_lengths = query_lengths if causal else [65, 8]
    cu_q = torch.tensor([0, 3, 9], dtype=torch.int32, device=device)
    cu_k = torch.tensor(
        [0, key_lengths[0], sum(key_lengths)], dtype=torch.int32, device=device
    )
    q = torch.randn(9, heads, 192, device=device, dtype=torch.bfloat16) * 0.1
    k = (
        torch.randn(sum(key_lengths), heads, 192, device=device, dtype=torch.bfloat16)
        * 0.1
    )
    # The projection's V view is interleaved with K and must be packed.
    kv = (
        torch.randn(sum(key_lengths), heads, 256, device=device, dtype=torch.bfloat16)
        * 0.1
    )
    v = kv[..., 128:]
    out = torch.empty(9, heads, 128, device=device, dtype=torch.bfloat16)
    metadata = SimpleNamespace(query_start_loc=cu_q, max_query_len=6)
    backend.prepare_metadata(metadata)
    chunk = SimpleNamespace(
        query_start_loc=cu_q,
        cu_seq_lens=cu_k,
        max_query_len=6,
        max_seq_len=max(key_lengths),
    )

    def run():
        if causal:
            return backend.run_prefill_new_tokens(q, k, v, True, out=out)
        return backend.run_prefill_context_chunk(chunk, q, k, v, out=out)

    def check(actual, lse):
        q_start = k_start = 0
        for nq, nk in zip(query_lengths, key_lengths):
            logits = (
                torch.einsum(
                    "qhd,khd->hqk",
                    q[q_start : q_start + nq].float(),
                    k[k_start : k_start + nk].float(),
                )
                * scale
            )
            if causal:
                mask = torch.ones(nq, nk, device=device, dtype=torch.bool).triu(1)
                logits.masked_fill_(mask, float("-inf"))
            expected = torch.einsum(
                "hqk,khd->qhd", logits.softmax(-1), v[k_start : k_start + nk].float()
            )
            torch.testing.assert_close(
                actual[q_start : q_start + nq].float(), expected, atol=5e-4, rtol=2e-2
            )
            torch.testing.assert_close(
                lse[:, q_start : q_start + nq],
                logits.logsumexp(-1),
                atol=2e-5,
                rtol=2e-5,
            )
            q_start += nq
            k_start += nk

    with PreparationSession(device=device, autotune=False) as session:
        session.prepare(tuple(r for unit in units for r in unit.requests))
        check(*run())
        manager.lock()
        session.freeze()
        graph = torch.cuda.CUDAGraph()
        try:
            with kernel_resolution_guard("prepared MLA prefill replay"):
                with session.capture(), torch.cuda.graph(graph):
                    actual, lse = run()
                q.neg_()
                kv.mul_(-0.5)
                graph.replay()
                check(actual, lse)
                assert actual.data_ptr() == out.data_ptr()
        finally:
            graph.reset()
