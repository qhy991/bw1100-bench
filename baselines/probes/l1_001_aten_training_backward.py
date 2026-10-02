"""Pending original GQA backward baseline using shipped ATen training ops.

The task supplies softmax weights and a dropout mask, so a whole q/k/v
FlashAttention backward is the wrong ABI. This composition uses PyTorch's
native dropout backward and fused softmax backward plus its vendor GEMMs;
layouts, GQA aggregation and output casts adapt to the unchanged task ABI.
No baseline qualification or speed claim is implied by this file's presence.
Use only the pinned DTK image and the canonical 16x10 original device gate.
"""

import torch


@torch.no_grad()
def run(grad_attn_output, attn_weights, attn_weights_dropped,
        value_states, dropout_mask, attention_dropout):
    batch, seq_q, heads, head_dim = grad_attn_output.shape
    kv_heads, seq_kv = value_states.shape[1:3]
    groups = heads // kv_heads
    values = value_states[:, :, None, :, :].expand(
        batch, kv_heads, groups, seq_kv, head_dim).reshape(
            batch, heads, seq_kv, head_dim)
    grad = grad_attn_output.transpose(1, 2).float()
    dprob = torch.matmul(grad, values.float().transpose(-2, -1))
    if float(attention_dropout) > 0:
        dprob = torch.ops.aten.native_dropout_backward.default(
            dprob, dropout_mask, 1.0 / (1.0 - float(attention_dropout)))
    dscores = torch.ops.aten._softmax_backward_data.default(
        dprob, attn_weights.float(), -1, torch.float32)
    dvalues = torch.matmul(attn_weights_dropped.float().transpose(-2, -1), grad)
    dvalues = dvalues.reshape(batch, kv_heads, groups, seq_kv, head_dim).sum(dim=2)
    return dscores.to(torch.bfloat16), dvalues.to(torch.bfloat16)
