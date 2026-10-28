"""fp64 oracle for the data2 KDA seam (Glm5NextLinearAttention._forward, bounded safe gate).

Plain PyTorch float64, no vLLM import: the candidate-side correctness oracle. Every rule below is
read from the kernels the served path runs, not inferred from shapes:

conv (`vllm/model_executor/layers/mamba/ops/causal_conv1d.py::_causal_conv1d_update_kernel`,
IS_SPEC_DECODING branch). The conv state of a request is ONE pool line (`spec_state_indices[:,0]`)
holding `state_len = (width-1) + (seqlen-1)` = 6 columns for a 4-token verify step, 3 for decode.
  offset = num_accepted - 1 (0 for decode)
  history = state[:, offset : offset+width-1]                      (3 columns)
  out_t   = silu( sum_w weight[c, w] * [history, x_0..x_{T-1}][t + w] )
  new state = [ state[:, offset+1], state[:, offset+2], x_0, .., x_{T-1} ]   (shift by one)
  written back to the SAME line (the 1-D index tensor makes the write target index 0).

recurrence (`vllm_hcu/.../patch_glm5next_kda_strided_recurrent.py`, COMPUTE_GATE + SAFE_GATE):
  g_t   = LOWER / (1 + exp(-exp(A_log[h]) * (g1[t,h,:] + dt_bias[h*D:(h+1)*D])))
  q,k L2-normalised per head with eps 1e-6; q scaled by K**-0.5; beta_t = sigmoid(beta_raw[t,h])
  State layout is [H, V, K] with K fastest (kernel address `i_hv*V*K + o_v[:,None]*K + o_k[None,:]`;
  K == V == 128 here, so a transposed reading is shape-legal and silently wrong).
  S  <- S * exp(g_t) broadcast over V     (decay on the K axis)
  v' <- beta_t * (v_t - S k_t)            (contract K)
  S  <- S + outer(v', k_t)
  o_t = S q_t                              (contract K)
  verify: initial S from ssm[slots[num_accepted-1]]; token t writes its S to ssm[slots[t]].
  decode: S from ssm[slot]; the single token writes back to the same slot.

`forward` returns (core_attn_out [M, H, D] fp64, conv_state, ssm_state) with the slots updated on
clones of the inputs.
"""

from __future__ import annotations

import torch

H, D, WIDTH, LOWER = 64, 128, 4, -5.0


def gate(g1: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor) -> torch.Tensor:
    """g1 [T, H, D] -> gate [T, H, D] in (LOWER, 0)."""
    g = g1.double()
    bias = dt_bias.double().reshape(H, D)
    a = torch.exp(a_log.double().reshape(H, 1))
    return LOWER / (1.0 + torch.exp(-(a * (g + bias))))


def conv_update(x: torch.Tensor, state: torch.Tensor, weight: torch.Tensor, offset: int):
    """x [T, C] tokens; state [C, state_len] this request's conv line; weight [C, WIDTH].
    Returns (silu out [T, C], new state [C, state_len])."""
    x = x.double().t()
    st = state.double()
    T = x.shape[1]
    seq = torch.cat([st[:, offset: offset + WIDTH - 1], x], dim=1)
    w = weight.double()
    out = torch.stack([(seq[:, t: t + WIDTH] * w).sum(1) for t in range(T)], dim=1)
    new = torch.cat([st[:, offset + 1: offset + WIDTH - 1], x], dim=1)
    if new.shape[1] < st.shape[1]:
        new = new[:, -st.shape[1]:]
    return torch.nn.functional.silu(out).t(), new[:, : st.shape[1]]


def l2n(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + eps)


def recurrence(q, k, v, g, beta, state):
    """q,k,v,g [T,H,D]; beta [T,H]; state [H,V,K] as stored. Returns (o [T,H,D], state after each token)."""
    S = state.clone()
    outs, per_token = [], []
    for t in range(q.shape[0]):
        S = S * torch.exp(g[t]).unsqueeze(-2)
        vp = v[t] - torch.einsum("hvk,hk->hv", S, k[t])
        vp = beta[t].unsqueeze(-1) * vp
        S = S + vp.unsqueeze(-1) * k[t].unsqueeze(-2)
        outs.append(torch.einsum("hvk,hk->hv", S, q[t]))
        per_token.append(S.clone())
    return torch.stack(outs), per_token


def forward(qkv, g1, beta_raw, a_log, dt_bias, conv_w, conv_state, ssm_state,
            slots_per_req, tokens_per_req, accepted=None):
    """One step. qkv [M, 3*H*D]; g1 [M,H,D]; beta_raw [M,H]; conv_state [L, C, state_len] (dim-major);
    ssm_state [L, H, D, D]; slots_per_req: slot ids per request (1 for decode, spec+1 for verify);
    tokens_per_req: query tokens per request; accepted: num_accepted per request (verify only)."""
    conv_state = conv_state.double().clone()
    ssm_state = ssm_state.double().clone()
    M = sum(tokens_per_req)
    out = torch.zeros(M, H, D, dtype=torch.float64)
    g_all = gate(g1[:M], a_log, dt_bias)
    b_all = torch.sigmoid(beta_raw[:M].double())
    cur = 0
    for r, slots in enumerate(slots_per_req):
        n = tokens_per_req[r]
        rows = slice(cur, cur + n)
        cur += n
        off = (int(accepted[r]) - 1) if accepted is not None else 0
        y, new_conv = conv_update(qkv[rows], conv_state[slots[0]], conv_w, off)
        conv_state[slots[0]] = new_conv
        q = l2n(y[:, : H * D].reshape(n, H, D)) * (D ** -0.5)
        k = l2n(y[:, H * D: 2 * H * D].reshape(n, H, D))
        v = y[:, 2 * H * D:].reshape(n, H, D)
        o, per_token = recurrence(q, k, v, g_all[rows], b_all[rows], ssm_state[slots[off]])
        out[rows] = o
        for t in range(n):
            ssm_state[slots[t if accepted is not None else 0]] = per_token[t]
    return out, conv_state, ssm_state
