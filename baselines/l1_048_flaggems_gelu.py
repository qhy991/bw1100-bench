"""FlagGems GELU-tanh-and-mul baseline for original L1/048.

The original title says SwiGLU, but its frozen reference computes two BF16
projections followed by GELU-tanh(gate) * up. This adapter calls the shipped
FlagGems 5.4.0dev fused activation from the pinned FlagRelease source while
keeping the original projection ABI. It was qualified on gfx938 using all 16
original workloads and ten fresh rounds; it is not a whole vLLM model result.
"""

import hashlib
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


_root = Path(__file__).resolve().parents[1]
_source = _root / ".deps/flag_gems_src_540_node2"
_path = _source / "flag_gems/fused/gelu_and_mul.py"
if hashlib.sha256(_path.read_bytes()).hexdigest() != (
        "5a21e5f2994c8a0b88d4676b99b1f1621891e55b1527ebe64d90f6ab7857a532"):
    raise RuntimeError("FlagGems GELU source differs from pinned FlagRelease image")
sys.path.insert(0, str(_source))
import flag_gems  # noqa: E402


if flag_gems.__version__ != "5.4.0dev" or flag_gems.vendor_name != "hygon":
    raise RuntimeError("Expected pinned FlagGems Hygon backend")


@torch.no_grad()
def run(x, gate_proj, up_proj):
    gate = F.linear(x, gate_proj)
    up = F.linear(x, up_proj)
    return flag_gems.gelu_and_mul(gate, up, approximate="tanh")
