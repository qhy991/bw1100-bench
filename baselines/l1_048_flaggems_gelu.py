"""FlagGems GELU-tanh followed by the original BF16 multiply for L1/048.

The original title says SwiGLU, but its frozen reference computes two BF16
projections followed by GELU-tanh(gate), a BF16 round, and multiplication by up.
This adapter calls the shipped Hygon GELU from pinned FlagGems 5.4.0dev. The
earlier fused activation omitted the intermediate round; its numerical pass
does not qualify this repaired baseline. Full gfx938 qualification is pending.
"""

import hashlib
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


_root = Path(__file__).resolve().parents[1]
_source = _root / ".deps/flag_gems_src_540_node2"
_path = _source / "flag_gems/runtime/backend/_hygon/ops/gelu.py"
if hashlib.sha256(_path.read_bytes()).hexdigest() != (
        "025ffb1a126d82d9850f310a099d2abfe9e648ba462176b6073aada999d035f1"):
    raise RuntimeError("FlagGems GELU source differs from pinned FlagRelease image")
sys.path.insert(0, str(_source))
import flag_gems  # noqa: E402
from flag_gems.runtime.backend._hygon.ops.gelu import gelu as _gelu  # noqa: E402


if flag_gems.__version__ != "5.4.0dev" or flag_gems.vendor_name != "hygon":
    raise RuntimeError("Expected pinned FlagGems Hygon backend")


@torch.no_grad()
def run(x, gate_proj, up_proj):
    gate = F.linear(x, gate_proj)
    up = F.linear(x, up_proj)
    activated_gate = _gelu(gate, approximate="tanh").to(gate.dtype)
    return activated_gate * up
