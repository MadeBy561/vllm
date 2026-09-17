# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HyperConnection (Gated Residual) utilities — NVIDIA model variant.

Implements the HyperConnection residual scheme proposed in
"HyperConnections" (https://arxiv.org/abs/2409.19606). This NVIDIA variant
delays each HC combine to the following HC mix boundary. HC glue kernels,
including fused combine+RMSNorm, live in ``ops/hc.py``; projections remain
standard vLLM Linear modules.

Hidden states between layers have shape ``[..., HC*HS]`` with HS inner
(HC outer, HS inner — checkpoint-native layout).

Typical usage inside a transformer decoder layer::

    self.attn_hc = GatedResidual(hc_config)

    hidden_states, block_input, injection = self.attn_hc.mix(hidden_states)
    attention_output = attention(block_input)
    hidden_states, block_input, injection = self.mlp_hc.combine_and_mix(
        hidden_states, attention_output, injection
    )
"""

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.models.utils import maybe_prefix
from vllm.platforms import current_platform
from vllm.utils.b12x import (
    B12xPreparationUnit,
    B12xWorkload,
    PreparationResourceUnavailableError,
    get_b12x_hyperconnection,
    set_b12x_preparation_provider,
)

from ..common.hyperconnection import (
    GroupedGemmaRMSNorm,
    HyperConnectionConfig,
)
from .ops.hc import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
    hc_gate_mix,
    hc_silu,
)


# ---------------------------------------------------------------------------
# Gated-residual variant
# ---------------------------------------------------------------------------
def _hyperconnection_api() -> Any:
    api = get_b12x_hyperconnection()
    if api is None:
        raise ImportError(
            "Qwen4Exp requires b12x.norm.hyperconnection; "
            "install the b12x serving extra"
        )
    return api


class HyperConnectionWorkspace(nn.Module):
    """Fixed-capacity storage shared by all HC modules in one model."""

    def __init__(self, config: HyperConnectionConfig, max_tokens: int) -> None:
        super().__init__()
        if not config.hc_per_branch_norm:
            raise NotImplementedError(
                "Qwen4Exp requires one RMSNorm group per HC stream"
            )
        self.config = config
        self.max_tokens = int(max_tokens)
        self.device = torch.device(current_platform.current_device())
        width = config.hc_count * config.hidden_size
        factory = dict(device=self.device, dtype=config.params_dtype)
        self.register_buffer(
            "normalized", torch.empty(max_tokens, width, **factory), persistent=False
        )
        self.register_buffer(
            "bottleneck",
            torch.empty(max_tokens, config.hc_lowrank, **factory),
            persistent=False,
        )
        self.register_buffer(
            "block_input",
            torch.empty(max_tokens, config.hidden_size, **factory),
            persistent=False,
        )

    def caps(self, max_tokens: int):
        api = _hyperconnection_api()
        return api.Caps(
            device=self.device,
            max_tokens=max_tokens,
            hidden_size=self.config.hidden_size,
            streams=self.config.hc_count,
            lowrank=self.config.hc_lowrank,
            dtype=self.config.params_dtype,
        )

    def bind(self, plan, tokens: int):
        return _hyperconnection_api().bind(
            plan,
            normalized=self.normalized,
            bottleneck=self.bottleneck,
            block_input=self.block_input,
            tokens=tokens,
        )


class GatedResidual(nn.Module):
    """Gated HyperConnection with learnable low-rank mixing and injection.

    ``combine_and_mix()`` runs the pre pipeline (grouped GemmaRMSNorm -> merged
    low-rank down+inject GEMM -> silu -> up GEMM -> sigmoid -> gated mean
    over the HC streams). When passed a pending block output, it fuses its
    residual combine with the RMSNorm. A missing injection selects unit-weight
    combine. Final mixers use ``use_combine=False`` and do not produce a new
    injection.

    Weights: the norm owns the grouped GemmaRMSNorm affine; the projections
    are vLLM Linear modules (merged replicated linear for down+inject), so
    GEMM dispatch (e.g. the low-latency skinny GEMM) applies through the
    standard quant_method mechanism.
    """

    def __init__(
        self,
        config: HyperConnectionConfig,
        use_combine: bool = True,
        prefix: str = "",
        *,
        workspace: HyperConnectionWorkspace | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.lora_rank = config.hc_lowrank
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.use_combine = use_combine

        norm_size = (
            self.hyper_hidden_size if config.hc_per_branch_norm else config.hidden_size
        )
        group_size = config.hidden_size if config.hc_per_branch_norm else None
        # Normalize each H-sized HC stream independently while retaining a
        # separate affine weight for every element of the HC*H layout.
        self.hc_norm = GroupedGemmaRMSNorm(
            norm_size,
            eps=config.rms_norm_eps,
            group_size=group_size,
            dtype=config.params_dtype,
        )

        # -- vLLM Linear weights --------------------------------------------
        # The merged skinny-GEMM shape is physically padded to 16 rows to ensure
        # good alignment and performant implementation chosen by CuBLAS heuristics.
        self.pad_size = (-(self.lora_rank + self.hc_count)) % 16 if use_combine else 0
        if use_combine:
            self.input_mix_weight_down_block_inject = MergedColumnParallelLinear(
                self.hyper_hidden_size,
                [self.lora_rank, self.hc_count]
                + ([self.pad_size] if self.pad_size else []),
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_down_block_inject"),
                return_bias=False,
                disable_tp=True,
            )
        else:
            self.input_mix_weight_down = ReplicatedLinear(
                self.hyper_hidden_size,
                self.lora_rank,
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_down"),
                return_bias=False,
            )
        self.input_mix_weight_up = ReplicatedLinear(
            self.lora_rank,
            self.hyper_hidden_size,
            bias=False,
            params_dtype=config.params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "input_mix_weight_up"),
            return_bias=False,
        )
        object.__setattr__(self, "_workspace", workspace)
        self._preparation_prefix = prefix or "qwen4_exp.hyperconnection"
        self._plans: dict[str, object] = {}
        if workspace is not None and not getattr(
            self, "b12x_preparation_suppressed", False
        ):
            set_b12x_preparation_provider(self, self)

    def mix(self, hidden_states: torch.Tensor):
        if self.workspace is not None:
            normalized = _hyperconnection_api().run_grouped_rmsnorm(
                hidden_states,
                self.hc_norm.weight,
                eps=self.config.rms_norm_eps,
                binding=self._binding(hidden_states, "grouped_rmsnorm"),
            )
        else:
            normalized = grouped_gemma_rmsnorm(
                hidden_states,
                self.hc_norm.weight,
                self.config.rms_norm_eps,
                self.hc_count,
            )
        block_input, injection = self._mix_normalized(normalized)
        return hidden_states, block_input, injection

    def combine_and_mix(self, hidden_states, prev_block_output, prev_injection):
        if prev_block_output is None:
            return self.mix(hidden_states)
        if self.workspace is not None and prev_injection is not None:
            combined, normalized = _hyperconnection_api().run_combine_norm(
                hidden_states,
                prev_block_output,
                prev_injection,
                self.hc_norm.weight,
                eps=self.config.rms_norm_eps,
                plan=self._plan_for("combine_norm"),
            )
        else:
            combined, normalized = hc_combine_norm(
                hidden_states,
                prev_block_output,
                prev_injection,
                self.hc_norm.weight,
                self.config.rms_norm_eps,
                self.hc_count,
            )
        block_input, injection = self._mix_normalized(normalized)
        return combined, block_input, injection

    def combine(self, hidden_states, block_output, injection):
        if self.workspace is not None and injection is not None:
            return _hyperconnection_api().run_combine(
                hidden_states,
                block_output,
                injection,
                plan=self._plan_for("combine"),
            )
        return hc_combine(hidden_states, block_output, injection, self.hc_count)

    @property
    def hyper_hidden_size(self) -> int:
        return self.hc_count * self.hidden_size

    def _request_name(self, operation: str, tokens: int) -> str:
        return f"{self._preparation_prefix}.hc.{operation}.m{tokens}"

    def get_b12x_preparation_units(
        self, layer: torch.nn.Module, workload: B12xWorkload
    ) -> Sequence[B12xPreparationUnit]:
        if layer is not self:
            raise ValueError("HC preparation owner mismatch")
        if self.workspace is None or workload.stage != "weights":
            return ()
        if self.hc_norm.weight.is_meta:
            return ()
        if workload.max_tokens > self.workspace.max_tokens:
            raise PreparationResourceUnavailableError(
                f"{self._preparation_prefix} HC workspace capacity "
                f"{self.workspace.max_tokens} cannot serve {workload.max_tokens}"
            )
        api = _hyperconnection_api()
        operations = (
            "grouped_rmsnorm",
            "scaled_silu",
            "gate_mean",
            "combine",
            "combine_norm",
        )
        requests = []
        tokens = workload.max_tokens
        plans = {}
        for operation in operations:
            plan = api.plan(
                self.workspace.caps(tokens),
                invocation={"operation": operation, "eps": self.config.rms_norm_eps},
            )
            plans[operation] = plan
            requests.append(
                plan.request(
                    name=self._request_name(operation, tokens),
                    prepare_call=self._prepare_call(operation, tokens),
                    benchmark_call=self._benchmark_call(operation, tokens),
                )
            )
        self._plans = plans
        return (
            B12xPreparationUnit(
                name="HYPERCONNECTION",
                key=(self._preparation_prefix, tokens),
                requests=tuple(requests),
                stage="weights",
            ),
        )

    def _prepare_call(self, operation: str, tokens: int):
        """Prime the installed operation against its durable workspace."""
        return self._call_factory(operation, tokens, benchmark=False)

    def _benchmark_call(self, operation: str, tokens: int):
        """Measure an isolated binding; it is never retained for serving."""
        return self._call_factory(operation, tokens, benchmark=True)

    def _call_factory(self, operation: str, tokens: int, *, benchmark: bool):
        def prepare(state):
            from b12x.norm.hyperconnection import _impl
            from b12x.preparation import PreparedCall

            factory = dict(device=self.workspace.device, dtype=self.config.params_dtype)
            width = self.hyper_hidden_size
            activation_inputs = []
            activation_owners: list[torch.Tensor] = []
            owners: tuple[torch.Tensor, ...]

            def activation(shape):
                value = torch.empty(shape, **factory)
                template = torch.arange(value.numel(), **factory).reshape(shape)
                template.div_(max(value.numel(), 1))
                activation_inputs.append((value, template))
                activation_owners.extend((value, template))
                return value

            def produce():
                for value, template in activation_inputs:
                    value.copy_(template)

            # Serving priming writes its owned workspace. Trials instead own
            # independent output storage so no measured binding can escape.
            def output(shape, serving):
                return torch.empty(shape, **factory) if benchmark else serving

            if operation == "grouped_rmsnorm":
                source = activation((tokens, width))
                out = output((tokens, width), self.workspace.normalized)
                run = lambda: _impl.run_grouped_rmsnorm_impl(
                    source,
                    self.hc_norm.weight,
                    eps=self.config.rms_norm_eps,
                    plan=state,
                    out=out,
                )
                owners = (out,)
            elif operation == "scaled_silu":
                source = activation((tokens, self.lora_rank))
                out = output((tokens, self.lora_rank), self.workspace.bottleneck)
                run = lambda: _impl.run_scaled_silu_impl(source, plan=state, out=out)
                owners = (out,)
            elif operation == "gate_mean":
                source = activation((tokens, width))
                gates = activation((tokens, width))
                out = output((tokens, self.hidden_size), self.workspace.block_input)
                run = lambda: _impl.run_gate_mean_impl(
                    source, gates, plan=state, out=out
                )
                owners = (out,)
            else:
                hidden = activation((tokens, width))
                block = activation((tokens, self.hidden_size))
                injection = activation((tokens, self.hc_count))
                if operation == "combine":
                    run = lambda: _impl.run_combine_impl(
                        hidden,
                        block,
                        injection,
                        plan=state,
                    )
                else:
                    run = lambda: _impl.run_combine_norm_impl(
                        hidden,
                        block,
                        injection,
                        self.hc_norm.weight,
                        eps=self.config.rms_norm_eps,
                        plan=state,
                    )
                owners = ()
            return PreparedCall(
                run=run,
                produce=produce,
                owners=(*activation_owners, *owners),
            )

        return prepare

    def _plan_for(self, operation: str):
        try:
            return self._plans[operation]
        except KeyError:
            raise PreparationResourceUnavailableError(
                f"{self._preparation_prefix} lacks a declared {operation} plan"
            ) from None

    @property
    def workspace(self) -> HyperConnectionWorkspace:
        return self._workspace

    def _binding(self, hidden_states: torch.Tensor, operation: str):
        return self.workspace.bind(self._plan_for(operation), hidden_states.shape[0])

    def _mix_normalized(self, normalized: torch.Tensor):
        api = _hyperconnection_api() if self.workspace is not None else None
        if self.use_combine:
            down_and_injection = self.input_mix_weight_down_block_inject(normalized)
            projected_down = down_and_injection[:, : self.lora_rank]
            injection_start = self.lora_rank
            # The projection owner stays live through the downstream residual
            # combine; readers consume row-strided slices without staging.
            injection = down_and_injection[
                :, injection_start : injection_start + self.hc_count
            ]
        else:
            projected_down = self.input_mix_weight_down(normalized)
            injection = None

        bottleneck = (
            api.run_scaled_silu(
                projected_down, binding=self._binding(normalized, "scaled_silu")
            )
            if api is not None
            else hc_silu(projected_down, self.hc_count)
        )
        gate_logits = self.input_mix_weight_up(bottleneck)
        block_input = (
            api.run_gate_mean(
                normalized, gate_logits, binding=self._binding(normalized, "gate_mean")
            )
            if api is not None
            else hc_gate_mix(normalized, gate_logits, self.hc_count)
        )
        return block_input, injection


__all__ = [
    "GatedResidual",
    "GroupedGemmaRMSNorm",
    "HyperConnectionConfig",
]
