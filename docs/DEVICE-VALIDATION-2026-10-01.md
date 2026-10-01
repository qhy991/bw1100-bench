# BW1100/BW1101 device-path validation, 2026-10-01

Evidence owner: the named local Cake admission receipts and benchmark result JSON
under the tested node's ignored `.local/` and `results/` directories. The upstream
benchmark source is `a9fa0804c793d438e70850c33fe34426e66d53dd` and the dataset
revision is `63699402f003496acc3af4eb534a5304a8ac1ea9`. The benchmark runner
source used for the first three accepted device checks is
`9b693e011be04df7da1c4f1b1bd8f9c103188c6e`. The final reusable-entry
smoke uses `3608a766318482167924f0425470827a801f6bdf`.

The first host had eight HCU cards at approximately 97% VRAM occupancy. Its
existing DTK image passed the 10-task/160-workload audit and L1/069 reference
and independent candidate CPU smoke. No GPU work was submitted to its occupied cards.

On the second host, seven `BW1101/gfx938` HCU cards showed zero VRAM usage and no KFD
processes before device work. Two available images passed the same CPU audit/candidate
checks. The chosen DTK image was the existing vLLM 0.29 image with Python 3.10.12,
Hygon PyTorch 2.11.0, Triton 3.6.0, and HIP 6.3.26113. Its local image identity was
`sha256:3ad0ae7192b8f9bafdf5b48fc414f8785f3c2463005e6b25290b7f75146ff260`.

## Exact outcomes

| Observation | Outcome | Authority |
| --- | --- | --- |
| L1/069 independent Torch candidate, original smoke workload, two rounds | 2/2 PASS, maximum absolute error 0; partial only | `results/l1-069-device-candidate-node4-smoke-007-gitfix-20261001.json`; Cake `hip-276bbbcabba0` |
| L1/069 independent Torch candidate, all 16 original workloads, ten fresh rounds each | 160/160 PASS, 0 failures, maximum absolute error 0, `full_device_correctness=true` for this one task | `results/l1-069-device-candidate-node4-full-001-20261001.json`; Cake `hip-c04947eada1b` |
| L2/060 original smoke workload, two rounds of reference selfcheck | 2/2 PASS; demonstrates the reference path runs on HCU, no independent candidate qualification | `results/l2-060-device-reference-smoke-001-20261001.json`; Cake `hip-32e9452d60c7` |
| Published `scripts/dtk.sh gpu` with `examples/l1_069_torch_baseline.py`, original L1/069 smoke workload, two rounds | 2/2 PASS, maximum absolute error 0; validates the reusable container/Cake launch entry, partial workload only | `results/l1-069-gpu-facade-20261001.json` and `results/l1-069-gpu-facade-cake-20261001-terminal.json`; Cake `hip-da0670085525` |

For every device attempt, the Cake Hygon `local_broker` issued a local-serialized
job id. It coordinates this container and child. It does not exclude external GPU
activity, so none of the results is a physical-exclusivity or performance claim.
KFD process checks after accepted runs reported no remaining compute PIDs; HCU VRAM
returned to 0% in the observed status snapshot.

## First divergence and repair

The first transient image used Hygon PyTorch 2.10.0. The current node's container
settings initially ran it under the ordinary host UID, and Torch failed before
any kernel call (`No HIP GPUs are available`). Removing `ROCR_VISIBLE_DEVICES=0`,
adding `/dev/mkfd`, and enabling the existing vLLM container's `privileged` setting
did not fix that ordinary-UID environment. In a subsequent read-only probe,
`rocminfo` exited 8 under ordinary UID despite readable device nodes. The same
PyTorch 2.11 image and flags under the existing vLLM container's root identity
enumerated one `BW1101/gfx938` device successfully. This localizes the observed
entrypoint failure to the container user identity within this image/host route;
it does not establish a generic Hygon requirement.

The first root-mode benchmark attempt then stopped before device execution at
Git's repository-ownership protection. The runner now scopes `safe.directory` to
its pinned SOL-ExecBench checkout while keeping exact revision and source-diff
checks. Ten CPU regression tests passed for that fix. The two accepted device
receipts use the repaired source. Earlier failed attempts and their terminal
Cake receipts remain in `.local/`.

No AITER operator, non-Torch optimized candidate, other eight suite tasks, model
serving, or performance timing was qualified by this record.
