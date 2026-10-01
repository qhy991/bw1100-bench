"""Candidate vLLM fused-MoE baseline for original L2/024 (unqualified).

The original ABI provides separate gate/up weights. vLLM's shipped fused
expert kernel expects them concatenated as ``w1``; this adapter packs only
that layout and passes the original routing IDs/weights to the community
kernel. Packing cost is part of this callable and must be disclosed in any
future latency comparison. The reference computes expert math in FP32, so
device correctness is required before this can be called a baseline.
"""

import hashlib
from pathlib import Path

import torch
import vllm
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts


if vllm.__version__ != "0.29.0":
    raise RuntimeError("Expected installed vLLM 0.29.0")
_source = (Path(vllm.__file__).resolve().parent /
           "model_executor/layers/fused_moe/fused_moe.py")
if hashlib.sha256(_source.read_bytes()).hexdigest() != (
        "7072eb06237be9d33dcb0ef7101410f886a6363c98cbee70a014c68b70f639cb"):
    raise RuntimeError("vLLM fused MoE source differs from pinned image")


@torch.no_grad()
def run(hidden_states, topk_indices, topk_weights,
        gate_proj_weights, up_proj_weights, down_proj_weights):
    w1 = torch.cat((gate_proj_weights, up_proj_weights), dim=1).contiguous()
    return fused_experts(
        hidden_states.contiguous(), w1, down_proj_weights.contiguous(),
        topk_weights, topk_indices,
    )
