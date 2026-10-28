# GLM-5.3 production operators

These tasks evaluate the data2 GLM-5.3-Flash local call on BW1100 `gfx938`. They are
additional selections in `suite.json.production_tasks`. The original SOL v2 twelve
tasks, upstream oracle, workloads and historical denominator are unchanged.

```
python bwbench.py production list
python bwbench.py production audit --task glm53-kda-verify --inputs .data/production/kda-data2
python bwbench.py production audit --task glm53-lmhead --model .data/production/model
```

`list` and `audit` use only Python's standard library. Audit checks manifests and
coverage; it does not validate tensor values or establish device correctness.
`check` requires the existing HCU admission path, the data2 vendor image with vLLM
`0.28.1+dtk2610`, Hygon HIP and an exact `gfx938` device. Use `scripts/dtk.sh gpu`
and its documented options, then run this command inside that admitted process:

```
python bwbench.py production check --task glm53-kda-verify \
  --inputs .data/production/kda-data2 --model .data/production/model \
  --candidate .local/candidate.py --device cuda:0 --output results/kda-check-001.json
```

The HCU wrapper mounts this checkout at `/work`. Materialize the external inputs
and required model files under ignored `.data/production/` before entering it.
A symlink to an unmounted host path is insufficient. Never commit captures,
checkpoint tensors or generated reports. Reports are create-only. Missing prerequisites are `blocked`; a rejected candidate or failed check is `failed`.
The report records the failure stage and whether device checking started. `performance`
is always `not_measured`: this revision provides correctness tooling, not a new
timing qualification, SOTA claim or serving acceptance.

## Candidate interface

A candidate module exports `prepare(case)`, which returns a zero-argument callable.
Preparation can derive reusable data from weights and structural metadata. The
call writes the complete seam into the Bench-owned `case['output']` and state
buffers. Input-dependent work belongs in that call. The Bench retains and checks
all input, weight, output and state tensors itself. A candidate cannot replace
buffers or provide its own integrity verdict.

The dictionary contains `inputs`, `weights`, `metadata`, `state` and `output`.
KDA uses `state['conv']` and `state['ssm']`; its remapped device metadata is in
`inputs` with the original `md_` prefixes. LM-head has `inputs['x']`,
`weights['weight']` and no mutable state. A complete Torch bring-up implementation
for the latter is:

```python
import torch


def prepare(case):
    x = case['inputs']['x']
    weight = case['weights']['weight']
    output = case['output']

    def call():
        torch.matmul(x, weight.t(), out=output)

    return call
```

This is a mixed/library implementation, not a Cake or owned-kernel result.
Candidates cannot import the native arm, oracle, vLLM or evaluator. The source
scan is a contract check, not a security sandbox. Verification runs independently
from authoring. Keep reference access as known-kernel reproduction when the
source or current native implementation has been disclosed.

## KDA verify

The seam is `Glm5NextLinearAttention._forward`: causal convolution update and
bounded gated-delta recurrence after merged projection, before `o_norm/o_proj`.
The task is one local TP1 call from a DP8/EP8 deployment. It contains no
communication. B1 with MTP k=3 has M=4 in the target verify step. M=8,16,32 are
required guards, not additional batch-size interpretations.

Inputs are the original BF16 strided qkv projection, g1 and beta views, prepared
conv weight, A_log and dt_bias, accepted-token metadata and state mappings.
The conv pool has SD layout `[slots,6,24576]` BF16. The recurrent state is
`[slots,64,128,128]` FP32 with `[H,V,K]` indexing. The output and every written
state row are judged. Slot zero and unused pool rows remain unchanged. Padded
output rows are undefined in the captured ABI and are outside the judged region.

Supply the original `kda-data2` directory containing both `rank0/manifest.json`
and `rank5/manifest.json`, all 48 `.pt` archives and their original native replay
admission. Rank0 contains requests 2,4,8; rank5 contains 1,2,4. Both layers 4 and
24 and four consecutive calls per structural case are required. The existing
manifest digest is checked once when each tensor archive crosses into the
verifier. The decoder preserves recorded offsets, strides, storage extent and aliasing,
with a one-GiB decoded-storage limit per alias group. No source campaign checkout is imported.

The native arm requires both served HCU patches. Each retained native case gets
its own receiver and unique prefix, even if two cases share prepared weights.
The checker first replays native twice against the admitted captured outputs.
It then checks candidate eager output, all written state, untouched regions,
input/weight integrity, retained outputs across later calls, changed-content
graph replay and graph chains with prior queued stream work. Actual chain lengths are reported.
All retained cases are checked before resets and after later graph work.

Acceptance in this revision is strict captured/native equality. The extracted
FP64 model is available in `production.kda_reference` for diagnosis. The source
cell's floating tolerance is not adopted as calibrated acceptance: its seeded
wrong-slot, gate and normalization counterexamples have not yet been qualified.
The current corpus has verify records; it does not qualify pure decode, prefill,
all accepted-token histories, model quality, MTP acceptance or full serving.

## DSA decode

The selected task is the complete indexer followed by sparse MLA. Indexer score,
top-k and index-cache updates belong inside it; MLA consumes that selection and
updates its paged `fp8_ds_mla` cache. The intended scored M is 4 with guards
8,16,32. Both stage timings must be reported when a future timing contract is
qualified. A fixed-top-k MLA subcase is not a full DSA result.

This revision refuses DSA `audit` and `check` before loading a candidate. The
source task still awaits complete real producer captures, fresh-process native
replay admission, independent oracle and evaluator. A fabricated manifest does
not enable execution. The capture must retain the real
`impl.topk_indices_buffer`, both stage boundaries, prepared indexer parameters,
cache read/write regions and unchanged-byte obligations. Freeze selection and
tie semantics before accepting indexer changes.

## LM-head

The task is complete BF16 logits `[M,4096] @ [154880,4096].T`, excluding sampling
and the separate draft local-argmax route. The weight is the real external
checkpoint's `lm_head.weight`. Activations are explicitly synthetic, using two
fixed seeds at M=4,8,16,32. The input tier remains in the catalog and report.

The evaluator compares all finite logits to FP64 with the source rule
`relative_error <= 1.05 * native_error + 2^-9`. Argmax must agree with FP64 on
every row where native agrees. It checks input/weight integrity, retained
outputs and changed-content graph replay. A pass under these conditions is
neither a production-input qualification nor an end-to-end model claim.

## Source and pending qualification

The source is the user-authored `glm53-b1-decode-20261027` package, local base
`1a054f6` with uncommitted capture-hook changes, reconciled with its remote
KDA/LM-head source at `f1f2829`. Only mathematical reference, archive decoding and
minimal adapter behavior were extracted. Active controllers, candidate kernels,
reservation logic, optimization budgets and historical results were excluded.

CPU tests verify tooling and refusal behavior. This change launches no capture,
HCU evaluation, experiment or serving process. Independent device execution of
the new evaluator and its graph/state obligations remains necessary before any
new device-correctness statement. Paired timing, precision calibration and
strong-reference screening remain separate prospective work.
