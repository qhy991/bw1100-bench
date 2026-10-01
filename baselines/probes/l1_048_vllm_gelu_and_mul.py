"""Candidate community baseline for original L1/048 (not yet qualified).

The installed vLLM `GeluAndMul(approximate="tanh")` supplies the GeGLU
activation after two BF16 projections. The task title mentions SwiGLU, but
the frozen reference computes GELU-tanh. This is a composite adapter, not a
claim of a single vLLM whole-task operator or a measured speed baseline.
"""

import torch
import torch.nn.functional as F
import vllm
from vllm.model_executor.layers.activation import GeluAndMul


if vllm.__version__ != "0.29.0":
    raise RuntimeError("Expected installed vLLM 0.29.0")

_activation = GeluAndMul(approximate="tanh")


@torch.no_grad()
def run(x, gate_proj, up_proj):
    gate = F.linear(x, gate_proj)
    up = F.linear(x, up_proj)
    return _activation(torch.cat((gate, up), dim=-1))
