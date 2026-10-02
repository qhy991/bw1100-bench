"""Pending mixed-gate community FLA composition with original FP32 dataflow.

Unlike the earlier probe, explicitly expand key heads in the original cyclic
order, pad all inputs before community kernels, retain FP32 intermediates,
and scale queries before dot products. Kernels are the unchanged vLLM-bundled
FLA entries. Require TRITON_F32_DEFAULT=ieee before import/compilation.
"""

import os
import torch
import torch.nn.functional as F

if os.environ.get('TRITON_F32_DEFAULT') != 'ieee':
    raise RuntimeError('Require explicit IEEE FP32 dot precision for this contract')

from vllm.third_party.flash_linear_attention.ops.cumsum import chunk_local_cumsum
from vllm.third_party.flash_linear_attention.ops.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from vllm.third_party.flash_linear_attention.ops.solve_tril import solve_tril
from vllm.third_party.flash_linear_attention.ops.wy_fast import recompute_w_u_fwd
from vllm.third_party.flash_linear_attention.ops.chunk_delta_h import chunk_gated_delta_rule_fwd_kernel_h_blockdim64
from vllm.third_party.flash_linear_attention.ops.chunk_o import chunk_fwd_o


def _state_fp32(k, w, u, raw):
    # The shipped wrapper tunes only stages 2/3; its FP32 launch exceeded
    # gfx938's 64 KiB LDS. Invoke the same unedited JIT body with one stage.
    b, t, heads, dk = k.shape
    dv = u.shape[-1]
    h = k.new_empty(b, t // 64, heads, dv, dk)
    new_v = torch.empty_like(u)
    jit = chunk_gated_delta_rule_fwd_kernel_h_blockdim64.fn.fn
    jit[((dv + 31) // 32, b * heads)](
        k=k, v=u, w=w, v_new=new_v, g=raw, gk=None, h=h,
        h0=None, ht=None, cu_seqlens=None, chunk_offsets=None,
        T=t, H=heads, Hg=heads, K=dk, V=dv, BT=64, BV=32,
        USE_G=True, USE_GK=False, USE_INITIAL_STATE=False,
        STORE_FINAL_STATE=False, SAVE_NEW_VALUE=True, IS_VARLEN=False,
        USE_EXP2=False, num_warps=4, num_stages=1)
    return h, new_v


@torch.no_grad()
def run(query, key, value, g, beta, scale):
    dtype = query.dtype
    b, hk, t, dk = key.shape
    hv, dv = value.shape[1], value.shape[-1]
    pad = (-t) % 64
    q = query * torch.rsqrt((query * query).sum(-1, keepdim=True) + 1e-6)
    k = key * torch.rsqrt((key * key).sum(-1, keepdim=True) + 1e-6)
    q = q.float() * float(scale)
    k, v, raw, update = k.float(), value.float(), g.float(), beta.float()
    # Reference expansion is cyclic [h0,h1,...,h0,h1,...], not interleaved.
    q = q.repeat(1, hv // hk, 1, 1)
    k = k.repeat(1, hv // hk, 1, 1)
    if pad:
        q, k, v = [F.pad(x, (0, 0, 0, pad)) for x in (q, k, v)]
        raw, update = [F.pad(x, (0, pad)) for x in (raw, update)]
    tp, nt = t + pad, (t + pad) // 64
    q, k, v = [x.transpose(1, 2).contiguous() for x in (q, k, v)]
    raw, update = [x.transpose(1, 2).contiguous() for x in (raw, update)]
    cumulative = chunk_local_cumsum(raw, 64)
    a = chunk_scaled_dot_kkt_fwd(k=k, g=cumulative, beta=update,
                                chunk_size=64, output_dtype=torch.float32)
    a = solve_tril(a, output_dtype=torch.float32)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=update, g_cumsum=raw,
                            A=a, cu_seqlens=None)
    h, new_v = _state_fp32(k, w, u, raw)
    out = chunk_fwd_o(q=q, k=k, v=new_v, h=h, g=cumulative,
                      scale=1.0, chunk_size=64)
    q_chunks = q.reshape(b, nt, 64, hv, dk).permute(0, 3, 1, 2, 4)
    state = h.reshape(b, nt, hv, dv, dk).permute(0, 2, 1, 4, 3)
    correction = torch.matmul(q_chunks, state)
    factors = (raw.exp() - cumulative.exp()).reshape(b, nt, 64, hv)
    correction = correction * factors.permute(0, 3, 1, 2).unsqueeze(-1)
    correction = correction.permute(0, 2, 3, 1, 4).reshape(b, tp, hv, dv)
    return (out.float() + correction)[:, :t].contiguous().to(dtype)
