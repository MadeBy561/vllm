# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Checkpoint padding for Kimi projections with whole heads per TP rank."""

from torch import nn


def enable_kimi_projection_tail_padding(layer: nn.Module) -> None:
    """Allow checkpoint-absent TP tails in explicitly padded Kimi projections."""
    for parameter in layer.parameters(recurse=False):
        parameter.allow_tp_padding = True
