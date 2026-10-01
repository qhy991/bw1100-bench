"""Candidate community composite for original L2/018 (not yet qualified).

Keeps the task's QKV projection, supplied RoPE inputs and final projection;
uses the installed FlashAttention varlen kernel for the ragged attention core.
The original reference rounds softmax probabilities to BF16 before its value
matmul, so this adapter must pass the original numeric gate before it can be
considered a baseline. A source import or a matching shape is insufficient.
"""

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func
from importlib.metadata import version


if version("flash_attn") != "2.8.3+dtk2604.torch2110.2608271707.g328942":
    raise RuntimeError("FlashAttention build differs from the pinned node4 image")


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


@torch.no_grad()
def run(hidden_states, cu_seqlens, cos, sin, qkv_weight, qkv_bias,
        proj_weight, proj_bias):
    total, embed_dim = hidden_states.shape
    heads, head_dim = cos.shape[1:]
    qkv = F.linear(hidden_states, qkv_weight, qkv_bias)
    q, k, v = qkv.reshape(total, 3, heads, head_dim).unbind(dim=1)
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    v = v.contiguous()

    cu = torch.cat((cu_seqlens.new_zeros(1), cu_seqlens)).to(torch.int32)
    lengths = cu[1:] - cu[:-1]
    max_len = int(lengths.max().item())
    if max_len == 0:
        return torch.zeros_like(hidden_states)
    attention = flash_attn_varlen_func(
        q, k, v, cu, cu, max_len, max_len,
        dropout_p=0.0, softmax_scale=head_dim ** -0.5, causal=False,
    )
    return F.linear(attention.reshape(total, embed_dim), proj_weight, proj_bias)
