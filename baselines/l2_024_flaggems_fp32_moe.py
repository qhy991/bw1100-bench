"""FlagGems FP32 grouped-GEMM MoE baseline for original L2/024.

The original reference executes the full MoE chain in FP32: it upcasts the
routed tokens, the three per-expert weight matrices and all intermediates
to FP32, computes SiLU(gate(x)) * up(x) @ down in FP32, applies the
softmax-normalized top-8 weights and scatter-adds in FP32, rounding only
the final output to BF16.

A CPU probe (``campaign/tools/l2_024_cpu_numeric_probe.py``) shows that even
an ideal BF16 fused-MoE kernel path (BF16 GEMMs with FP32 accumulation,
FP32 SiLU, FP32 combine) fails the original smoke gate: matched ratio
0.96946 vs the raw 0.98 / effective 0.99 requirement, max_abs 1.562e-2 vs
atol 5.1e-4. Therefore shipped BF16 fused MoE kernels (vLLM
``fused_experts``, AITER ``fused_moe`` — whose earlier attempt also hit an
HCU memory-aperture VMFault — or FlagGems fused MoE) cannot satisfy the
reference semantics regardless of layout fixes.

This adapter preserves the reference semantics exactly by composing the
shipped FlagGems 5.4.0dev (Hygon) Triton ``bmm`` primitive (true FP32
``tl.dot`` with ``allow_tf32=False``) for the three per-expert GEMMs,
while keeping the original routing/dispatch (one-hot mask, ``where``,
gather), SiLU, weighting, scatter-add and final BF16 rounding on the
framework path.

Classification: minimal composition of shipped community primitives
(FlagGems bmm), not a fused MoE kernel and not a relabeled vLLM/AITER
route.
"""

import hashlib
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


_root = Path(__file__).resolve().parents[1]
_source = _root / ".deps/flag_gems_src_540_node2"
_path = _source / "flag_gems/ops/bmm.py"
if hashlib.sha256(_path.read_bytes()).hexdigest() != (
        "e50e4463a28092cd18d5e15dc372068f6b484e7416268c2e0133dd4724cfbec7"):
    raise RuntimeError("FlagGems bmm source differs from pinned FlagRelease image")
sys.path.insert(0, str(_source))
import flag_gems  # noqa: E402


if flag_gems.__version__ != "5.4.0dev" or flag_gems.vendor_name != "hygon":
    raise RuntimeError("Expected pinned FlagGems Hygon backend")


def _mm_fp32(x2d, w_t):
    """FP32 matmul through the shipped FlagGems bmm kernel (batch 1)."""
    return flag_gems.bmm(x2d.unsqueeze(0), w_t.unsqueeze(0)).squeeze(0)


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_proj_weights: torch.Tensor,
    up_proj_weights: torch.Tensor,
    down_proj_weights: torch.Tensor,
):
    num_tokens = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    n_routed_experts = gate_proj_weights.shape[0]

    final_hidden_states = torch.zeros(
        num_tokens, hidden_size, dtype=torch.float32, device=hidden_states.device
    )

    expert_mask = F.one_hot(topk_indices, num_classes=n_routed_experts)
    expert_mask = expert_mask.permute(2, 0, 1)  # [n_experts, num_tokens, k]

    for expert_idx in range(n_routed_experts):
        mask = expert_mask[expert_idx]  # [num_tokens, k]

        token_indices, weight_indices = torch.where(mask)

        if token_indices.numel() > 0:
            expert_input = hidden_states[token_indices].to(torch.float32)

            expert_weights = topk_weights[token_indices, weight_indices].to(torch.float32)

            gate_w = gate_proj_weights[expert_idx].to(torch.float32)
            up_w = up_proj_weights[expert_idx].to(torch.float32)
            down_w = down_proj_weights[expert_idx].to(torch.float32)

            # SwiGLU: down(silu(gate(x)) * up(x)), all FP32 like the reference.
            gate_output = _mm_fp32(expert_input, gate_w.t())
            gate_output = F.silu(gate_output)

            up_output = _mm_fp32(expert_input, up_w.t())

            intermediate = gate_output * up_output

            expert_output = _mm_fp32(intermediate, down_w.t())

            weighted_output = expert_output * expert_weights.unsqueeze(-1)

            final_hidden_states.index_add_(0, token_indices, weighted_output)

    return final_hidden_states.to(torch.bfloat16)
