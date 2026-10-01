"""Community fused baseline adapter for L1/069 residual+RMSNorm.

Pins the installed Hygon vLLM 0.29.0 aiter path: the `ir.ops.fused_add_rms_norm`
"aiter" impl (`vllm.kernels.aiter_ops.fused_add_rms_norm`, which calls
`torch.ops.vllm_aiter.fused_add_rms_norm` -> `aiter.rmsnorm2d_fwd_with_add`).
This is exactly the kernel vLLM serving dispatches on AMD/Hygon ROCm when
`VLLM_ROCM_USE_AITER=1` (see vllm/platforms/rocm.py:1115-1128). The impl is
invoked through the same registered impl entry vLLM's dispatcher uses, only
with the priority override made explicit instead of env-driven.

Image: vllm0.29.0-ubuntu22.04-dtk26.04-py3.10-20260831-qwen3.8flashnext
(image_id sha256:3ad0ae7192b8f9bafdf5b48fc414f8785f3c2463005e6b25290b7f75146ff260),
kernels/aiter_ops.py sha256 256cbc4e9fc27baac3f0640f3f97a73e4df0da113459cd7e138eee502dda70e2,
ir/ops/layernorm.py sha256 65d33dcb96404ddde273acf84ef901151a8155a2cffc144bdd0c49fe1d576a22.

Adapter only: no algorithmic changes. Out-of-place (the aiter impl allocates
new output tensors and does not mutate inputs).
"""
import os

import torch
import vllm
import vllm.kernels  # noqa: F401  (registers the IR kernel impls used below)
from vllm import ir

# The gateway container has a read-only root filesystem; aiter's jit core
# would copy its jit dir to the read-only ~/.aiter at import time.
# AITER_JIT_DIR redirects that to the container's writable tmpfs.
os.environ.setdefault("AITER_JIT_DIR", "/tmp/bwbench-aiter-jit")

_IMPL = ir.ops.fused_add_rms_norm.impls["aiter"].impl_fn


@torch.no_grad()
def run(hidden_states: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    out, _ = _IMPL(hidden_states, residual, weight, float(eps))
    return out
