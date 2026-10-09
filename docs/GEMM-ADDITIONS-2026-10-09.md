# Two GEMM additions

Suite revision 2 appends two original SOL-ExecBench L1 tasks. The first ten task
records, their smoke IDs, seed 200, ten correctness rounds, target gfx938, source
revision and dataset revision remain unchanged. The extended suite has twelve
tasks, 192 original workloads and 1920 correctness cells for a complete run.

| Task | Precision | GEMM dimensions | Original workloads | Smoke |
| --- | --- | --- | ---: | --- |
| L1/003 LM head projection | BF16 inputs and output | M=batch×sequence, K=2048, N=102400; M=128..8192 | 16 | batch=1, sequence=128 |
| L1/077 Whisper output projection | FP16 inputs and output | M=batch×sequence, K=1280, N=51866; M=1..39232 | 16 | batch=1, sequence=1 |

Both original references compute `hidden_states @ weight.T`. Their public input
ABI is two tensors, and their output is a tensor retaining the batch and sequence
dimensions. They provide standalone GEMM coverage alongside the existing fused
GateUp, MoE and attention tasks. L1/003 tests a large, aligned vocabulary dimension;
L1/077 adds FP16, decode-like M=1 cases and a vocabulary dimension with tile tails.
Each includes the original irregular sequence lengths. No shape is reduced or
generated to replace an upstream workload.

Despite the L1/003 title, all sixteen original workloads have
`logits_to_keep == seq_len`. The reference computes every sequence position and
the ABI supplies no slicing scalar. The baseline therefore returns the full
projection; introducing an extra slicing policy would change the task.

The largest L1/003 output is about 1.56 GiB, and the largest L1/077 output is
about 3.79 GiB. Device checks also need reference and candidate outputs, input
copies, and library workspace; arrange them through the existing bounded gateway.

## Sources and acceptance

The task rows come from dataset revision
`63699402f003496acc3af4eb534a5304a8ac1ea9`, paired with SOL-ExecBench source
`a9fa0804c793d438e70850c33fe34426e66d53dd`, as recorded by `sources.lock.json`.
Definitions, references and all workloads are materialized only in ignored
`.data/`. The original NVIDIA Evaluation Dataset License remains applicable.
Revision 2 records preparation in `.data/materialization-suite-v2.json`, preserving
the original ten-task `.data/materialization.json` when extending an existing checkout.

`baselines/l1_003_torch_lm_head.py` and
`baselines/l1_077_torch_whisper_output.py` are PyTorch/HIP vendor-GEMM baseline
candidates. They keep the original return-value ABI and dtype and do not cache
inputs or results. CPU interface checks are not gfx938 qualification. A strong
performance denominator requires the existing all-workload, ten-round device
gate, then the separately specified paired timing and A/A controls. There is no
speed result attached to these additions.

## Addition verification

Local preparation and `bwbench.py audit` completed for all twelve tasks and
192 workloads. The original ten task records and correctness settings were
checked against the predecessor suite and are unchanged. The repository's
eighteen CPU regression tests passed.

Two real original-workload CPU smoke attempts were bounded to 180 seconds for
L1/003 and 120 seconds for L1/077. Both timed out before creating a numerical
result report. They provide no numerical-pass evidence and were not repeated.
No GPU admission, candidate optimization, or baseline performance measurement
was performed as part of adding these tasks.

The already-running ten-task optimization/export snapshots remain ten-task
records. The two additions become available through the revision-2 suite for
subsequent qualification and optimization; old result aggregates are not relabeled
as twelve-task results.
