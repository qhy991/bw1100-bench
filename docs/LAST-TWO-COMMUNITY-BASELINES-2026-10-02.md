# Last two community baseline qualifications, 2026-10-02

These entries preserve the original task IDs, 16 workload grids, ten fresh
rounds, data generator and effective numeric/exact-discrete gates. They are
compositions of real installed community kernels, not claims that a whole
shipped MoE/FLA serving operator has the unusual benchmark ABI.

## L1/058: native stable sort and native lower bounds

`baselines/l1_058_aten_stable_search.py:run` calls the shipped PyTorch 2.11
HIP stable sort. That produces both expert values and the stable permutation.
The already sorted values feed the shipped `aten.searchsorted.Tensor` with
257 integer keys and `out_int32=True`, providing exact expert offsets without
a histogram/prefix-sum pipeline. The permutation is cast to the original
int32 output type. No sorting/counting GPU kernel was reimplemented.

This route passed original CPU smoke 2/2, GPU smoke 2/2
(`bw-2b295077acfe`), and all 16 workloads x 10 fresh rounds, 160/160
(`bw-fc1204b55857`). The final file including the histogram comparison arm
was rebound by 160/160, `bw-36dd0ee4c6fd`. Both full admissions issued
`completed`, exit 0.
Raw reports are in the ignored `results/l1-058-native-search-*` files under
node4's successor `experiments/bw1100-bench-last-two-20261002/`.
The old HCU0 FlagGems argsort timeout was not rerun or reclassified.

The core sort is the same optimized vendor-library primitive the oracle
already used; changing a wrapper name is not a strength claim. The new
offset algorithm is an actual different library composition. Paired
comparison with the independently full-qualified native histogram/prefix-sum
arm (`bw-6f1adbfc525b`) completed as `bw-22f76b25cdd6`: search was faster
in all 16 cells in both measurement orders, 2.07–2.37x, above A/A drift and
the 1% threshold. Inputs were unmutated. This is a community-arm screening
result, not a custom-candidate speed claim or universal fastest performance.

## L2/060: exact-contract FP32 FLA component composition

The original combines cumulative intra-chunk gates with raw per-token
inter-chunk gates, so its semantics differ from the fused standard GDN entry.
`baselines/l2_060_fla_fp32_mixed.py` uses unchanged vLLM 0.29.0 bundled FLA
chunk kernels with per-site gate arguments plus a vendor matmul correction.

The old adapter's diagnostic was premature: it gave standard interleaved
GVA head order to an original contract with cyclic head expansion, retained
BF16 intermediate allocations/triangular inverse despite the original FP32
dataflow, and used a ragged path instead of the original zero-padded chunk
inputs. The successor explicitly expands q/k cyclically to the value-head
count, pads all inputs to chunk 64, keeps FP32 intermediates and scales q
before dot products. `TRITON_F32_DEFAULT=ieee` and
`FLA_TRIL_PRECISION=ieee` are explicit per-command requirements.

After those corrections, the first device attempt failed at the state kernel
with shared memory Required 73728 / Hardware limit 65536, job
`bw-7b149678c55f`. The shipped wrapper tunes only stages 2/3. The successor
launches the same unedited JIT body with BV=32, num_warps=4, num_stages=1
and the same algorithm flags, fitting gfx938's LDS limit. No community
kernel source is modified.

The corrected smoke passed 2/2, job `bw-a70f7073254e`; the full original gate
passed 160/160 as job `bw-3aeba0c00011`, admission `completed`, exit 0.
Raw reports are `results/l2-060-fla-fp32-{smoke-002,full-001}.json` and their
admission/terminal companions in the same successor worktree. Failed attempts
remain create-only artifacts.

The unchanged state-kernel execution was independently collected as job
`bw-a98d5164c8e0` in `.local/profile/fla-final-smoke-002/`: the CSV verifier
accepted four matching `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`
dispatches, the requested Wavefronts/FETCH_SIZE/GPUBusy columns and hashes;
the admission ended completed/released. This proves the route executes the
community state body; it is not a no-profiler latency score or a complete
bottleneck diagnosis. A failed harness quoting attempt is retained separately.

The older statement that the shipped FLA kernels are generally numerically
broken on this hardware is withdrawn: the unchanged kernels complete this
original workload grid with appropriate semantic, dtype and launch bindings.
This does not prove the unmodified fused single-gate entry matches the original,
nor isolate one changed parameter as the sole cause of the older divergence.

```bash
HIP_VISIBLE_DEVICES=2 BWBENCH_TIMEOUT=1800 bash scripts/dtk.sh gpu IMAGE \
  results/UNIQUE-admission.json \
  env TRITON_F32_DEFAULT=ieee FLA_TRIL_PRECISION=ieee \
      TRITON_CACHE_DIR=/work/.local/triton-ieee \
  python3 bwbench.py check \
  --task L2/060_chunk_gated_delta_rule_linear_attention --device cuda:0 \
  --candidate /work/baselines/l2_060_fla_fp32_mixed.py \
  --workloads all --rounds 10 --output results/UNIQUE-check.json
```

Substitute the pinned installed image and fresh receipt/output paths. Keep
IEEE flags and persistent cache identical for profiling and any paired timing.
The image, source hashes and canonical qualification rows belong to the
baseline inventory; raw data/results stay outside Git.
