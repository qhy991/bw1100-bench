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

The original ten entries have a full original-workload gfx938 qualification.
Suite revision 2 also adds two standalone GEMM baseline candidates below; their
GPU qualification is pending and they have no performance result. The
table distinguishes shipped whole callables from compositions of optimized
community primitives; qualification is not universal performance optimality.
Final top-level entries L2/018, L2/024 and L2/056 were re-qualified after
moving them from the Ralph campaign (import-root paths had changed).
The final five-entry source and report binding is in
[the completion manifest](../docs/COMMUNITY-COMPLETION-2026-10-02.json).
The new last-two route and the withdrawn earlier FLA diagnosis are explained
in [last-two qualification](../docs/LAST-TWO-COMMUNITY-BASELINES-2026-10-02.md).

L2/060 requires these **per-command** settings before import/compilation:
`TRITON_F32_DEFAULT=ieee FLA_TRIL_PRECISION=ieee`, with a persistent writable
`TRITON_CACHE_DIR`. Keep them identical in correctness, profiling and timing;
do not silently use a different precision path. No installed library source
or host-wide environment setting was modified to qualify it.

The working node4 image is the existing DTK vLLM 0.29.0 image with immutable ID
`sha256:3ad0ae7192b8f9bafdf5b48fc414f8785f3c2463005e6b25290b7f75146ff260`.
It includes AITER 0.1.5, FlashAttention 2.8.3, and timm 1.0.28. The FlagOS
image on node2 is a **different** vLLM 0.26.1 / vllm-plugin-FL environment.
Only its FlagGems 5.4.0dev Python source was copied into node4's ignored
`.deps/` directory, pinned by `scripts/prepare_flag_gems.py`, and run in the
node4 image. This is a FlagGems operator baseline, **not** a full FlagOS-vLLM
model or plugin performance comparison.

| Task | Community path considered | Current classification | Evidence or missing gate |
| --- | --- | --- | --- |
| L1/069 residual RMSNorm | vLLM `fused_add_rms_norm` AITER implementation | **Full device correctness qualified** | `l1_069_vllm_aiter_fused.py` passed the original 16 workloads × 10 rounds, 160/160, on gfx938 through this repository's standalone runner. The alternative `vllm_c` path was rejected in the separate Ralph experiment: in-place ABI and a GPU no-op. No new paired latency was measured in this branch. |
| L1/011 Llama3 RoPE cos/sin output | Transformers 5.16.1 shipped `LlamaRotaryEmbedding`, eager / Inductor | **Full device correctness qualified; community arms screened** | `l1_011_transformers_rope.py` binds the original supplied frequency buffer to the shipped forward, then stacks its outputs. Both arms passed 160/160; the actual public-size dispatcher also passed 160/160. Paired wall timing selected compiled for 13 original cells and eager for the three larger cells, with dispatch cost included. This replaces the earlier handwritten vLLM cache reconstruction as the community denominator; it is not a global fastest-RoPE claim. |
| L1/048 dual GEMM + GELU-tanh gate | FlagGems `gelu_tanh_and_mul` after two original projections | **Full device correctness qualified** | `l1_048_flaggems_gelu.py` passed the original 16 workloads × 10 rounds on HCU1, 160/160, max_abs `0.0625`. Its FlagGems component is community fused; the whole adapter retains two framework GEMMs and has no paired performance qualification. The installed vLLM CustomOp route was rejected: its serving config was absent and `_C.gelu_tanh_and_mul` unregistered. Actual reference uses GELU-tanh, not title's SwiGLU. |
| L1/058 stable expert bucketing | PyTorch 2.11 HIP stable sort + native `searchsorted` | **Full device correctness qualified; native-library strength screened** | `l1_058_aten_stable_search.py` uses the sorted expert values to compute 257 exact lower bounds, preserving the exact stable int32 permutation and offsets. Final source passed 160/160. Against an independently qualified native histogram/prefix-sum library composition, search was faster in all 16 cells in both orders, 2.07–2.37x with A/A controls. The quarantined FlagGems argsort route was not reused. |
| L1/001 GQA attention backward | PyTorch 2.11 ATen native dropout / softmax backward and vendor GEMMs | **Full device correctness qualified; optimized community primitive composition** | `l1_001_aten_training_backward.py` uses shipped training backward operators on the supplied weights/mask, plus original GQA layouts/reduction. It passed 160/160 on HCU2 with a completed/released terminal receipt. This is a composition of optimized community primitives, not a whole vLLM backward callable; no speedup has been measured for a candidate against it. |
| L2/035 ConvNeXtV2 + GRN | timm 1.0.28 `ConvNeXtBlock(use_grn=True)` | **Full device correctness qualified** | `l2_035_timm_convnextv2.py` binds supplied weights through `torch.func.functional_call`. The original 16 workloads × 10 rounds passed 160/160 on gfx938, max_abs `1.86e-5`; this is a community block baseline for correctness, with no paired performance result yet. |
| L2/018 ragged vision attention | FlagGems materialized `bmm` + FP32 `softmax` composition | **Full device correctness qualified; community primitive composition** | `l2_018_flaggems_materialized.py` composes the shipped FlagGems 5.4.0dev Hygon Triton primitives with the reference's exact BF16 rounding points: the scaled score matrix is a BF16 `bmm` output (rounded before softmax) and FP32 softmax probabilities are rounded to BF16 before the BF16 PV `bmm`. The FlashAttention varlen arm was rejected because its FP32 online-softmax scores skip the first rounding (CPU probe max_abs 3.9e-3 vs smoke atol 3.1e-4; GPU smoke 3.05e-3). Passed the original 16 workloads × 10 rounds, 160/160 on HCU2 job `bw-76878b9d2628`, terminal completed/released. |
| L2/024 256-expert MoE | FlagGems FP32 grouped-GEMM composition | **Full device correctness qualified; community primitive composition** | `l2_024_flaggems_fp32_moe.py` keeps the original FP32 dispatch/SwiGLU/combine semantics and replaces the three per-expert GEMMs with the shipped FlagGems 5.4.0dev Hygon Triton `bmm` (true FP32 `tl.dot`). A CPU probe (`campaign/tools/l2_024_cpu_numeric_probe.py`) shows even an ideal BF16 fused kernel fails (matched 0.96946 < 0.98, max_abs 1.56e-2 vs atol 5.1e-4), so shipped BF16 fused MoE routes (vLLM `fused_experts`, AITER `fused_moe`, whose earlier attempt also VMFaulted) cannot meet the reference semantics. Passed 16 workloads × 10 rounds, 160/160 on HCU2 job `bw-4f5ca3ad565e`, terminal completed/released; max_abs 0.015625 is one BF16 ulp at the final rounding boundary within the effective upstream matched-ratio gate. |
| L2/060 chunk gated delta rule | Unchanged vLLM-bundled FLA component kernels, FP32 mixed-gate composition | **Full device correctness qualified; community component composition** | `l2_060_fla_fp32_mixed.py` retains cyclic heads, original zero padding and FP32 intermediates, routes gates per site, and uses IEEE dot precision. The unchanged state JIT body is launched at BV=32/warps=4/stages=1 to fit the 64 KiB LDS limit. Passed 160/160 and completed/released. Four actual state-kernel dispatches were validated by rocprof on the original smoke. This supersedes the old adapter-level device-negative diagnosis; the fused single-gate entry still has different semantics. |
| L2/056 complete decoder backward | PyTorch 2.11 ATen native training backward operators + vendor GEMMs | **Full device correctness qualified; community primitive composition** | `l2_056_aten_decoder_backward.py` binds the ten original gradient outputs and the original layouts/reduction chain, replacing the manual softmax gradient with `aten._softmax_backward_data` (FP32 operands, reference rounding point) and the manual SiLU derivative with `aten.silu_backward`, following the qualified L1/001 composition precedent. vLLM inference has no corresponding whole backward callable. Passed 16 workloads × 10 rounds, 160/160 on HCU2 job `bw-9105f0750a04`, terminal completed/released. |
| L1/003 BF16 LM head projection | PyTorch/HIP vendor GEMM (`torch.matmul`) | **Candidate prepared; gfx938 qualification pending** | `l1_003_torch_lm_head.py` preserves K=2048, N=102400 and all 16 original workloads. All original keep counts equal the sequence length; no extra slicing is introduced. CPU checks are not a device gate or speed result. |
| L1/077 FP16 Whisper output projection | PyTorch/HIP vendor GEMM (`torch.matmul`) | **Candidate prepared; gfx938 qualification pending** | `l1_077_torch_whisper_output.py` preserves K=1280, N=51866, decode M=1 and all 16 original workloads. CPU checks are not a device gate or speed result. |

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

The new L1/011 denominator is bound to source SHA-256
`8650dc499db88b902b75c5d2a17414ed39e2efb56dc6e1400ec7bb567deb498d`.
Its actual `run` full gate is job `bw-036e2d281a44`, report
`results/l1-011-transformers-public-full-002.json` (SHA-256
`eea24fed2eab1746829f4f5d8daab9b186d806c65b7abf27f44e6b3186eb23d2`),
terminal `results/l1-011-transformers-public-full-admission-002-terminal.json`
(SHA-256 `6584be3c7563f30c2bbea4b9f62ebafe178e974adb1d29a43e00a0ec470eb047`).
The final community-arm/dispatcher timing is job `bw-b772f2322074`, report
`results/l1-011-transformers-public-strength-003.json` (SHA-256
`a0c9af24b2699cb4cc900dd57d992e44e3ee851f19a74cc3a6b7506bc1d6d4e5`).
Original inputs, forward/reverse order and A/A controls were retained;
the direct dispatcher/selected-arm ratio ranged 0.9984–1.0118, median 1.0024.
These receipts live in node4's isolated
`experiments/bw1100-bench-baseline-qualification-20261002/`, not in Git.
See [RoPE qualification](../docs/ROPE-COMMUNITY-QUALIFICATION-2026-10-02.md)
for source/API binding, compiler failure history and selection scope. The
historical three-task Ralph RoPE ratio retains its old adapter denominator;
it is not silently converted into a Transformers-relative gain.

The L1/001 composition is bound to source SHA-256
`51a9006618dab0f1a90ea306b4bdcfa6b135e524b4d172d3b48852b7e023de34`
and job `bw-0d7d3de305b8`. In the same qualification worktree, full report
`results/l1-001-aten-full-002.json` has SHA-256
`c5155ac4d0140a0e03a0044c69fb80a4ba0ed21d4cadace099920678dee6d90d`;
terminal `results/l1-001-aten-full-admission-002-terminal.json` has SHA-256
`ab7530e6d90d32f770712bfefce678673b55f6cc9e58fc338ec82be6847114c0`.
The prior full numerical pass (`bw-02e8482a8963`) remained not-qualified
because the one-second post-exit sample still showed 32% VRAM. The new bounded
release observation recorded 32% -> 7% -> 0% in 2.78 seconds, with no KFD
process or live owned container, before issuing `completed`. This did not
change a numeric tolerance or reclassify the historical terminal receipt.
See [native backward qualification](../docs/ATEN-GQA-BACKWARD-QUALIFICATION-2026-10-02.md).

The L1/048 qualification is bound to source SHA-256
`637a1479ff71bad1d312c6da5d6ca51452298f99d27b89b7c7f5a9325d3eabd8`
and HCU1 job `bw-9ddbe6bc84ec`. The ignored raw report is
`results/l1-048-flaggems-top-hcu1-full-001.json` (SHA-256
`52ad5a9a6e81285091ae8d7a268f01e8c487ad9b353c403612e778e25aabc717`);
its terminal receipt is
`results/l1-048-flaggems-top-hcu1-full-admission-001-terminal.json`
(SHA-256 `54dc7b88020794f2da0a1c9586a4dfe2445900ab9d73bcbcbf3519bd0a76c932`).
After commit `03640a6` was applied to node4's primary checkout, its exact
runner and top-level adapter passed a fresh 2/2 HCU1 smoke (job
`bw-15cae0a76049`, terminal status `completed`, exit 0, HCU1 0% VRAM).
The original FlagGems source archive SHA-256 is
`b06d741b4d1f2978a539c38ced0ec41d1621455e5dec38b6350e5c9943dcda39`;
node4 retains it at `.deps/archives/flag_gems_src_540_node2.clean.tar.gz`.
It was exported from node2 FlagRelease image
`sha256:f06ff2697e5d84b90a52c4717702bd1e46c057881d78546b92e7ef4b5bbcddc6`
(`/workspace/FlagGems/src`), without committing that third-party source.
Run `python3 scripts/prepare_flag_gems.py --archive PATH` before using this
adapter in another checkout. The script verifies both the archive and the
materialized source; neither is committed.

The L2/018 qualification is bound to source SHA-256
`14052f456d5ef6d1d2df887731eedd16fb8d2817d95c6fdcfa405a62d1a4802e`
(`campaign/baselines/l2_018_flaggems_materialized.py`; the promoted copy in
this directory differs only in its repo-root path resolution). The ignored
raw full report is `campaign/results/l2-018-flaggems-full-001.json`
(SHA-256 `1fa6a5755ec650ec2cffe4f71642255b28a4cc2999fab6e2e9927c1b05e31213`,
160/160 cases, max_abs at most 0.001953125 under the effective upstream
gates); its terminal receipt is
`campaign/results/l2-018-flaggems-full-admission-001-terminal.json`
(SHA-256 `c361acf74f1f254a1d54d4fd28746b2e656057dd5a6ac9bd5c37739be245dd09`,
job `bw-76878b9d2628`, exit 0, HCU2 VRAM released to 0%). Smoke receipt:
job `bw-3af6b5fa6d8d` (report
`campaign/results/l2-018-flaggems-smoke-002.json`, SHA-256
`bfdad04e9cd2bc02a08da4e4d7d16d16ee0c925584a62c751e2c827b57ff8efc`).
The CPU divergence probe binding the earliest numeric split is
`campaign/tools/l2_018_cpu_divergence_probe.py` (SHA-256
`8c22322b05b859f25b9e54866cf3011fc2c69063ead484f55dd5112d6d1782ec`).
No speed claim is attached; no rocprof diagnostic was collected because
this phase claims no GPU bottleneck or strength mechanism for the entry
(`not_decision_relevant`).

The L2/024 qualification is bound to source SHA-256
`e7d6631a032f1c0feabaf3832eec655ae8dc996d29f44a7de37546f01aa16294`
(`campaign/baselines/l2_024_flaggems_fp32_moe.py`; the promoted copy in this
directory differs only in its repo-root path resolution). Full report
`campaign/results/l2-024-flaggems-full-001.json` (SHA-256
`aa4406c3733b70eb8e54cbcce1b4aeb00e2c6812542e94c939a25fd83ec1cf89`, 160/160)
and terminal `campaign/results/l2-024-flaggems-full-admission-001-terminal.json`
(SHA-256 `7fd8579664da77d1a9b3037175653ae8f2133a827f2bed889bbc401ad082ebc2`,
job `bw-4f5ca3ad565e`, exit 0, HCU2 VRAM 0%). Smoke: job `bw-98b97d73f242`
(report `campaign/results/l2-024-flaggems-smoke-001.json`, SHA-256
`da7f5f86f269e78898778080594fed66fa6370645f5e00bb047e263318550bf6`).
The BF16-fused infeasibility probe is
`campaign/tools/l2_024_cpu_numeric_probe.py` (SHA-256
`17d7dc2fe180211e8c80030def8d323c5273c928433792578586e1bea84f6bb7`).
No speed claim; rocprof is `not_decision_relevant` for this
correctness-only qualification.

The L2/056 qualification is bound to source SHA-256
`7af2d03e361cab0144c156907a882a8aeffaac739ccb22fd2b24974fb2e08236`
(identical file promoted to this directory; it has no path-relative
dependency). Full report `campaign/results/l2-056-aten-full-001.json`
(SHA-256 `6026cf8d66bdba6b564e9e3305ab6d4b89640d958f122eac89125d3d1657d6e5`,
160/160) and terminal
`campaign/results/l2-056-aten-full-admission-001-terminal.json` (SHA-256
`e0c0f8d6386baf1eac027d7ee8a606c82f748fec88de4427e46cb10e481056f8`, job
`bw-9105f0750a04`, exit 0, HCU2 VRAM 0%). Smoke: job `bw-c3ae129dcb0c`
(report `campaign/results/l2-056-aten-smoke-001.json`, SHA-256
`34c1c262db3b75c09c23de0b022d3f0a995d3aa41c99d1abbc39e9af181bb7a7`).
A CPU sanity run (`campaign/results/l2-056-aten-cpu-sanity-001.json`) was
started but CPU BLAS was too slow for the bounded phase; the container
exits on its own without holding any HCU. No speed claim; rocprof is
`not_decision_relevant` for this correctness-only qualification.

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

Only the top-level Python files here are baseline entries. Earlier failed
vLLM/FlashAttention probe source remains recoverable in experiment commit
`91820c5`; the L1/058 FlagGems probe is retained in the isolated node4
worktree under `baselines/probes/`. Create-only reports and terminal receipts
remain in that worktree's ignored `results/`.
Crash-prone probe code is not shipped as an active baseline. `examples/`
remains a simple Torch correctness candidate. Neither a pending row nor an
adapter/rejected row can support a strong-community speedup claim. This
inventory preserves gaps without changing task IDs, original dimensions, or
reference semantics.

The L1/058 timeout was a resource incident: job `bw-3ef7da817fac` has
terminal status `not_qualified`/124, but its named container remained live on
HCU0 at the recorded post-timeout snapshots. Ordinary `docker stop` and
`docker kill` were denied by the root:docker `hcu.sock`; no passwordless sudo
exists. Do not infer that HCU0 is now released from this historical record:
check the live container and HCU/KFD state, and use it only after privileged
cleanup is observed. Subsequent bounded checks used observed-idle HCU1 only.
The replacement inner timeout's HCU1 selfcheck (`bw-a86a4f64181d`) exited
124 and confirmed its own container removed.
