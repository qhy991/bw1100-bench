# Community baseline inventory

`bwbench.py` compares any supplied `run(*inputs)` candidate against the
source-locked SOL-ExecBench reference. The reference is the correctness oracle;
it is **not** an optimized performance baseline. This directory records
community implementations for the ten unchanged tasks. An installed library,
similar operator name, CPU pass, or adapted formula is not enough to claim an
equivalent strong baseline. Keep the original 16 workloads and ten fresh rounds
per task; only an exact-ABI gfx938 full gate can qualify a candidate baseline.
Latency additionally needs the same original generated inputs, source hashes,
absolute times, paired forward/reverse order, and A/A drift controls.

The working node4 image is the existing DTK vLLM 0.29.0 image with immutable ID
`sha256:3ad0ae7192b8f9bafdf5b48fc414f8785f3c2463005e6b25290b7f75146ff260`.
It includes AITER 0.1.5, FlashAttention 2.8.3, and timm 1.0.28. The FlagOS
image on node2 is a **different** vLLM 0.26.1 / vllm-plugin-FL environment;
it has not been qualified as a baseline on the free node4 HCU.

| Task | Community path considered | Current classification | Evidence or missing gate |
| --- | --- | --- | --- |
| L1/069 residual RMSNorm | vLLM `fused_add_rms_norm` AITER implementation | Direct community fused path | `l1_069_vllm_aiter_fused.py` is copied from the separate Ralph experiment. There it passed 16×10 on gfx938; this repository entry still needs its own fresh check. The alternative `vllm_c` path was rejected in that image: in-place ABI and a GPU no-op. |
| L1/011 Llama3 RoPE cos/sin output | vLLM rotary cache construction and gather | **Adapter only; no strong baseline** | No installed vLLM rotary callable returns the original `(B,S,D,2)` cos/sin ABI; the separate Ralph adapter reconstructs the pattern. Its measured ratio cannot be labeled a win against a direct community implementation. |
| L1/048 dual GEMM + GELU-tanh gate | vLLM `GeluAndMul(approximate="tanh")` after the two projections | Composite community candidate, unverified | `l1_048_vllm_gelu_and_mul.py` invokes the installed activation; its Hygon dispatch, BF16 rounding and full original ABI have not passed a device gate. The task's title says SwiGLU; the actual reference uses GELU-tanh. |
| L1/058 stable expert bucketing | AITER/vLLM MoE sorting or alignment | Semantic fit unverified | MoE alignment uses padded/token-block outputs; original task requires exact stable permutation and 257 expert offsets. A name match is insufficient. |
| L1/001 GQA attention backward | FlashAttention or training-attention backward | No exact callable identified | This task starts from already-materialized softmax weights and dropout mask and returns two specific gradients. vLLM is an inference stack; a q/k/v attention-backward call is not this ABI. |
| L2/035 ConvNeXtV2 + GRN | timm 1.0.28 `ConvNeXtBlock(use_grn=True)` | Direct community block candidate, unverified | `l2_035_timm_convnextv2.py` binds the supplied weights through `torch.func.functional_call` with the installed timm source pinned. CPU and gfx938 original-workload gates remain. |
| L2/018 ragged vision attention | FlashAttention varlen attention as one component | Composite community candidate, unverified | `l2_018_flashattn_varlen.py` keeps the original projections/RoPE and invokes the installed varlen kernel. BF16 softmax rounding and head_dim 72 still require original-task device correctness; a varlen kernel alone is not a whole-task baseline. |
| L2/024 256-expert MoE | vLLM/AITER fused MoE | Candidate path, unverified | Need to prove routing, FP32 expert math, weighted aggregation and output rounding against all original workloads; the largest input is resource-heavy. |
| L2/060 chunk gated delta rule | Bundled FLA `chunk_gated_delta_rule` | **Rejected for this task** | Separate Ralph experiment failed the first original smoke workload (max_abs ≈9.738): the reference and FLA use different inter-chunk gate conventions. A reference-semantics candidate passed correctness, but there is no qualifying community speed baseline. |
| L2/056 complete decoder backward | Training-model community backward components | No exact callable identified | Original ABI returns ten gradients with a particular intermediate/rounding chain. vLLM inference has no corresponding whole backward callable. |

Only code under this directory is a candidate community baseline. `examples/`
remains a simple independent Torch correctness candidate. Neither a pending
row nor an adapter/rejected row can be used as the denominator of a strong
community speedup claim. The baseline inventory preserves gaps instead of
changing task IDs, original dimensions, or reference semantics.
