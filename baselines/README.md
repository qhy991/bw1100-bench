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
| L1/069 residual RMSNorm | vLLM `fused_add_rms_norm` AITER implementation | **Full device correctness qualified** | `l1_069_vllm_aiter_fused.py` passed the original 16 workloads × 10 rounds, 160/160, on gfx938 through this repository's standalone runner. The alternative `vllm_c` path was rejected in the separate Ralph experiment: in-place ABI and a GPU no-op. No new paired latency was measured in this branch. |
| L1/011 Llama3 RoPE cos/sin output | vLLM rotary cache construction and gather | **Adapter only; no strong baseline** | No installed vLLM rotary callable returns the original `(B,S,D,2)` cos/sin ABI; the separate Ralph adapter reconstructs the pattern. Its measured ratio cannot be labeled a win against a direct community implementation. |
| L1/048 dual GEMM + GELU-tanh gate | vLLM `GeluAndMul(approximate="tanh")` after the two projections | **Rejected in current image** | `probes/l1_048_vllm_gelu_and_mul.py` stopped before device comparison because vLLM CustomOp needs a serving config. Direct inspection found `torch.ops._C.gelu_tanh_and_mul` unregistered and `vllm._rocm_C` absent. No fused community path was qualified; the task's actual reference uses GELU-tanh, not the title's SwiGLU. |
| L1/058 stable expert bucketing | AITER/vLLM MoE sorting or alignment | Semantic fit unverified | MoE alignment uses padded/token-block outputs; original task requires exact stable permutation and 257 expert offsets. A name match is insufficient. |
| L1/001 GQA attention backward | FlashAttention or training-attention backward | No exact callable identified | This task starts from already-materialized softmax weights and dropout mask and returns two specific gradients. vLLM is an inference stack; a q/k/v attention-backward call is not this ABI. |
| L2/035 ConvNeXtV2 + GRN | timm 1.0.28 `ConvNeXtBlock(use_grn=True)` | **Full device correctness qualified** | `l2_035_timm_convnextv2.py` binds supplied weights through `torch.func.functional_call`. The original 16 workloads × 10 rounds passed 160/160 on gfx938, max_abs `1.86e-5`; this is a community block baseline for correctness, with no paired performance result yet. |
| L2/018 ragged vision attention | FlashAttention varlen attention as one component | **Rejected at original smoke** | `probes/l2_018_flashattn_varlen.py` retained original projections/RoPE and called the installed varlen kernel. First original workload failed the upstream numeric gate (max_abs `0.00305` versus atol `0.00031`); BF16 softmax-probability rounding differs from the reference. |
| L2/024 256-expert MoE | vLLM `fused_experts` | **Runtime fault; not qualified** | `probes/l2_024_vllm_fused_moe.py` packed the original gate/up weights and invoked the installed fused kernel. The first original smoke hit an HCU memory-aperture VMFault and exited 139 before any comparison. The container ended and HCU VRAM later returned to 0%; do not repeat this route without a separately justified diagnosis. |
| L2/060 chunk gated delta rule | Bundled FLA `chunk_gated_delta_rule` | **Rejected for this task** | Separate Ralph experiment failed the first original smoke workload (max_abs ≈9.738): the reference and FLA use different inter-chunk gate conventions. A reference-semantics candidate passed correctness, but there is no qualifying community speed baseline. |
| L2/056 complete decoder backward | Training-model community backward components | No exact callable identified | Original ABI returns ten gradients with a particular intermediate/rounding chain. vLLM inference has no corresponding whole backward callable. |

The L1/069 qualification is bound to source SHA-256
`1430ad1e28a40be334b795de13083caa537a03049d92b5148530b2bee63c7b60`
and node4 job `bw-0a8e51e23d04`. The ignored raw report is
`results/l1-069-aiter-baseline-v2-full-001.json` (SHA-256
`bb5f543f3b4e1c795b9646cc95509ee00bf15ec6106c836c4dc23e576602ae17`);
its terminal receipt is
`results/l1-069-aiter-baseline-v2-full-admission-001-terminal.json`
(SHA-256 `9e2fc98f74354855c9c216db626b650bf7f70e26725d9caa03940b2c4bf0d5f1`).
The result is correctness-only; AITER JIT startup is not a measured speed
comparison, and the separate Ralph campaign's AITER-relative timing has a
different candidate/source binding.

The L2/035 qualification is bound to source SHA-256
`2cfe9f33fee5b7ac3bdbdcc31834bdb7101e78cc726a1eac71a7ab9b05b44d2c`
and node4 job `bw-31f7acf73a43`. The ignored raw report is
`results/l2-035-timm-gpu-full-001.json` (SHA-256
`211be2b677427719bd94571ccd22eb5099f0d178462f55ad5a13a2aa83e1dc9c`);
its terminal receipt is `results/l2-035-timm-gpu-full-admission-001-terminal.json`
(SHA-256 `73809a3a264b2e2f18eba1e2cfe3996f71945e85392f2cbd55b95d182faa8a4d`).
The first attempt failed in the *reference* convolution because MIOpen tried
to write `/root/.config/miopen` on a read-only root filesystem; rerunning with
`HOME=/tmp` passed smoke and full. The gateway now sets that writable HOME.

Top-level Python files are the only baseline entry candidates. `probes/`
preserves failed approaches for diagnosis and must not be used as a speed
denominator. `examples/` remains a simple Torch correctness candidate. Neither
a pending row nor an adapter/rejected row can support a strong-community
speedup claim. This inventory preserves gaps without changing task IDs,
original dimensions, or reference semantics.
