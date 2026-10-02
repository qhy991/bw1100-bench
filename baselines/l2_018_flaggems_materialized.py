"""FlagGems materialized varlen attention baseline for original L2/018.

The original reference computes, per ragged sequence, a *materialized*
attention path with two BF16 rounding points that online-softmax flash
kernels do not reproduce (CPU probe
``campaign/tools/l2_018_cpu_divergence_probe.py``):

1. the scaled score matrix is the BF16 output of a BF16 matmul
   (rounded before the FP32 softmax);
2. the FP32 softmax probabilities are rounded back to BF16 before the
   BF16 PV matmul.

The installed FlashAttention 2.8.3 varlen adapter failed the original
smoke at max_abs 0.00305 vs atol 0.00031 because it keeps QK^T scores
in FP32 through the online softmax. This adapter instead composes
shipped FlagGems 5.4.0dev (Hygon) primitives with the same rounding
points as the reference: FlagGems ``bmm`` (Triton ``tl.dot`` with FP32
accumulation, BF16 output) for both matmuls and FlagGems ``softmax``
computed in FP32 (``half_to_float`` output) then cast to BF16. The QKV
projection, RoPE, output projection and cu_seqlens splitting follow the
original reference ABI verbatim.

Classification: minimal composition of shipped community primitives
(FlagGems bmm + softmax), not a flash attention kernel and not a
hand-written reconstruction of the softmax chain.
"""

import hashlib
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from typing import Tuple


_root = Path(__file__).resolve().parents[1]
_source = _root / ".deps/flag_gems_src_540_node2"
_hashes = {
    "ops/bmm.py": "e50e4463a28092cd18d5e15dc372068f6b484e7416268c2e0133dd4724cfbec7",
    "ops/softmax.py": "d5233ee510bff1c3a9257058f1e4f6c6c8a4908615a7d96aaf91202ae8ac269a",
}
for _name, _expected in _hashes.items():
    _path = _source / "flag_gems" / _name
    if hashlib.sha256(_path.read_bytes()).hexdigest() != _expected:
        raise RuntimeError("FlagGems source differs: " + _name)
sys.path.insert(0, str(_source))
import flag_gems  # noqa: E402


if flag_gems.__version__ != "5.4.0dev" or flag_gems.vendor_name != "hygon":
    raise RuntimeError("Expected pinned FlagGems Hygon backend")


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_bias: torch.Tensor,
) -> torch.Tensor:
    embed_dim = 1152
    num_heads = 16
    head_dim = 72
    scaling = head_dim ** -0.5

    seq_length = hidden_states.shape[0]

    qkv = F.linear(hidden_states, qkv_weight, qkv_bias)
    qkv = qkv.reshape(seq_length, 3, num_heads, head_dim)
    query_states, key_states, value_states = qkv.permute(1, 0, 2, 3).unbind(0)

    query_states, key_states = _apply_rotary_pos_emb(
        query_states, key_states, cos, sin
    )

    cu_seqlens_with_zero = torch.cat([
        torch.zeros(1, dtype=torch.int64, device=cu_seqlens.device), cu_seqlens
    ])
    lengths = (cu_seqlens_with_zero[1:] - cu_seqlens_with_zero[:-1]).tolist()
    lengths = [l for l in lengths if l > 0]

    if len(lengths) == 0:
        return torch.zeros(
            seq_length, embed_dim, device=hidden_states.device, dtype=hidden_states.dtype
        )

    query_splits = torch.split(query_states, lengths, dim=0)
    key_splits = torch.split(key_states, lengths, dim=0)
    value_splits = torch.split(value_states, lengths, dim=0)

    attn_outputs = []
    for q, k, v in zip(query_splits, key_splits, value_splits):
        # [num_heads, seq_len_i, head_dim]
        qh = q.transpose(0, 1)
        kh = k.transpose(0, 1)
        vh = v.transpose(0, 1)

        # BF16 score matrix (FP32 accumulation, BF16 output) — same
        # rounding point as the reference's torch.matmul.
        attn_weights = flag_gems.bmm(qh, kh.transpose(1, 2)) * scaling
        # FP32 softmax over the BF16 scores, then BF16 probabilities.
        attn_weights = flag_gems.softmax(
            attn_weights, -1, half_to_float=True
        ).to(qh.dtype)
        # BF16 PV matmul with FP32 accumulation.
        attn_output = flag_gems.bmm(attn_weights, vh)

        attn_outputs.append(attn_output.transpose(0, 1))

    attn_output = torch.cat(attn_outputs, dim=0)
    attn_output = attn_output.reshape(seq_length, embed_dim)
    attn_output = F.linear(attn_output, proj_weight, proj_bias)
    return attn_output
