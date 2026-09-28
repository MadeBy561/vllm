# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint padding for Kimi projections with whole heads per TP rank."""

import torch
from torch import nn


def enable_kimi_projection_tail_padding(layer: nn.Module) -> None:
    """Allow checkpoint-absent TP tails in explicitly padded Kimi projections."""
    for parameter in layer.parameters(recurse=False):
        parameter.allow_tp_padding = True


def projection_checkpoint_dimensions(
    projections: dict[str, tuple[int, int]],
) -> dict[str, tuple[int | None, int]]:
    """Describe unpadded weight axes and MXFP8 scales grouped along K."""
    dimensions: dict[str, tuple[int | None, int]] = {}
    for name, (axis, size) in projections.items():
        dimensions[f"{name}.weight"] = (axis, size)
        dimensions[f"{name}.weight_scale"] = (
            axis,
            (size + 31) // 32 if axis == 1 else size,
        )
    return dimensions


def validate_checkpoint_tensor(
    name: str,
    tensor: torch.Tensor,
    dimensions: dict[str, tuple[int | None, int]],
) -> None:
    """Reject missing checkpoint heads before TP padding can conceal them."""
    expected = dimensions.get(name)
    if expected is None:
        return
    axis, size = expected
    actual = (
        tensor.numel()
        if axis is None
        else (tensor.shape[axis] if tensor.ndim > axis else -1)
    )
    if actual != size:
        raise ValueError(
            f"Kimi checkpoint tensor {name!r} has shape {tuple(tensor.shape)}; "
            f"expected {'element count' if axis is None else f'axis {axis}'} "
            f"{size} from the original head count, got {actual}."
        )
