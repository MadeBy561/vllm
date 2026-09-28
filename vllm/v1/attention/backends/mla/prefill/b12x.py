# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepared SM12x B12X MLA prefill with capacity-bounded packed sequences."""

import copy

import torch

from vllm.utils.b12x import B12xPreparationUnit
from vllm.v1.attention.backends.mla.prefill.base import MLADimensions, MLAPrefillBackend


class B12xPrefillBackend(MLAPrefillBackend):
    _prepared_causal_modes: tuple[bool, ...] = (True, False)
    supported_mla_dimensions = [
        MLADimensions(qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128)
    ]

    @staticmethod
    def get_name():
        return "B12X"

    @classmethod
    def supports_compute_capability(cls, device_capability):
        return device_capability.major == 12

    @classmethod
    def is_available(cls):
        try:
            from b12x.attention import varlen
        except ImportError:
            return False
        return varlen.is_supported()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._plans = {}

    def clone(self):
        # Plans contain immutable geometry, not request metadata or workspace.
        # Metadata builders share plans but assign their own prefill metadata.
        result = copy.copy(self)
        if hasattr(result, "_prefill_metadata"):
            del result._prefill_metadata
        return result

    def supports_out(self):
        return True

    def get_b12x_preparation_units(self, layer, workload):
        from b12x.attention import varlen

        from vllm.model_executor.layers.attention.mla_attention import (
            MLACommonMetadataBuilder,
        )

        if workload.stage != "weights":
            return ()
        rows = self.vllm_config.scheduler_config.max_num_batched_tokens
        seqs = self.vllm_config.scheduler_config.max_num_seqs
        kv_rows = MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size(
            self.vllm_config
        )
        dtype = self.vllm_config.model_config.dtype
        device = torch.device("cuda", torch.accelerator.current_device_index())
        qdim = self.qk_nope_head_dim + self.qk_rope_head_dim
        requests = []
        for causal in self._prepared_causal_modes:
            capacity = rows if causal else kv_rows
            if causal not in self._plans:
                # Planning consumes tensor metadata only; these temporary inputs
                # are not retained by the plan or duplicated per serving layer.
                q = torch.empty(
                    (rows, self.num_heads, qdim), device=device, dtype=dtype
                )
                k = torch.empty(
                    (capacity, self.num_heads, qdim), device=device, dtype=dtype
                )
                v = torch.empty(
                    (capacity, self.num_heads, self.v_head_dim),
                    device=device,
                    dtype=dtype,
                )
                cu = torch.empty(seqs + 1, device=device, dtype=torch.int32)
                self._plans[causal] = varlen.plan(
                    q,
                    k,
                    v,
                    cu,
                    cu,
                    max_seqlen_q=rows,
                    max_seqlen_k=capacity,
                    causal=causal,
                )
            role = "causal" if causal else "context"
            requests.append(
                self._plans[causal].request(
                    name=f"{layer.layer_name}.mla_prefill.{role}",
                    prepare_call=self._prepare_call,
                )
            )
        return (
            B12xPreparationUnit(
                name="B12X MLA prefill",
                key=(id(layer), "prefill"),
                requests=tuple(requests),
                stage="weights",
                autotune=False,
            ),
        )

    def _workspace(self, state):
        from vllm.v1.worker.workspace import current_workspace_manager

        (spec,) = state.scratch_plan.scratch_specs()
        # kv_b_proj emits interleaved K/V heads. The contiguous attention API
        # consumes a compact V view; this copy is included in prefill timings.
        specs = [(spec.shape, spec.dtype), (state.plan.v_shape, state.plan.dtype)]
        return current_workspace_manager().get_simultaneous(*specs)

    def _prepare_call(self, state):
        from b12x.preparation import PreparedCall

        scratch, values, *_ = self._workspace(state)
        q = torch.zeros(
            (1, *state.plan.q_shape[1:]),
            device=state.plan.device,
            dtype=state.plan.dtype,
        )
        k = torch.zeros(
            (1, *state.plan.k_shape[1:]),
            device=state.plan.device,
            dtype=state.plan.dtype,
        )
        values[:1].zero_()
        cu = torch.tensor([0, 1], device=state.plan.device, dtype=torch.int32)
        binding = state.bind(
            scratch=scratch,
            q=q,
            k=k,
            v=values[:1],
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=1,
            max_seqlen_k=1,
            softmax_scale=self.scale,
        )
        return PreparedCall(
            run=lambda: state.run(binding), owners=(q, k, cu, scratch, values, binding)
        )

    def _run(self, q, k, v, cu_q, cu_k, max_q, max_k, causal, out):
        from b12x.attention import varlen
        from b12x.preparation import require_prepared

        if causal not in self._plans:
            raise RuntimeError("B12X MLA prefill requires eager capacity preparation")
        plan = self._plans[causal]
        state = require_prepared(plan, "attention.varlen", q.device)
        scratch, values, *_ = self._workspace(state)
        if v.shape[0] > values.shape[0]:
            raise ValueError("MLA prefill KV rows exceed the prepared capacity")
        values = values[: v.shape[0]]
        if v.data_ptr() != values.data_ptr() or v.stride() != values.stride():
            values.copy_(v)
        binding = varlen.bind(
            plan,
            scratch=scratch,
            q=q,
            k=k,
            v=values,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            softmax_scale=self.scale,
        )
        output, lse = varlen.run(binding)
        # The suffix partial must survive subsequent context calls, which reuse
        # caller-owned scratch. Returned tensors therefore own their storage.
        if out is None:
            output = output.clone()
        else:
            out.copy_(output)
            output = out
        return output, lse.clone()

    def run_prefill_new_tokens(
        self, q, k, v, return_softmax_lse, out=None, output_scale=None
    ):
        if output_scale is not None:
            raise ValueError("B12X MLA prefill does not support quantized output")
        metadata = self._prefill_metadata
        output, lse = self._run(
            q,
            k,
            v,
            metadata.query_start_loc,
            metadata.query_start_loc,
            metadata.max_query_len,
            metadata.max_query_len,
            True,
            out,
        )
        return (output, lse) if return_softmax_lse else output

    def run_prefill_context_chunk(self, chunk, q, k, v, out=None):
        return self._run(
            q,
            k,
            v,
            chunk.query_start_loc,
            chunk.cu_seq_lens,
            chunk.max_query_len,
            chunk.max_seq_len,
            False,
            out,
        )
