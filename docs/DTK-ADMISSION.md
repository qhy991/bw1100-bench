# BW1100 standalone DTK/HCU admission

The active GPU entry is `scripts/dtk.sh gpu`, implemented by this repository's
`scripts/hcu_run.py`. It uses only the selected installed DTK image and local
system tools. It imports or mounts no code from another research project.
CPU mode stays a transient `runc` container without an HCU.

For each GPU command, the host runner takes a nonblocking, per-user file lock
at `~/.local/state/bw1100-bench/hcu-N.lock`, then records the selected HCU's
`hy-smi` VRAM percentage and whether `/dev/kfd` has a visible process. It
refuses to start unless the selected device reports 0% VRAM and no KFD PID is
visible. The image is inspected and its immutable ID recorded. A transient,
read-only-rootfs DTK container runs the exact command on the chosen HCU, with
network disabled, a writable `/tmp`, read-only hyhal mount and the benchmark
repository mounted at `/work`. The child has a bounded timeout, default 180 s;
set `BWBENCH_TIMEOUT` for an explicitly larger case (maximum 3600 s).

The output path is create-only. The initial `<name>.json` records admission,
source command, image, device and deadline; `<name>-terminal.json` records exit
code, container presence and post-run HCU/KFD observations. `completed` is
issued only for a zero exit and an observed released device. A missing terminal
or disconnected SSH is **unknown**, not a pass or an idle declaration: inspect
the exact container, HCU and artifacts before any resubmission.

The lock covers this user's suite runs only. Other users and containers can
still use the hardware, so receipts explicitly retain
`external_gpu_activity=not_excluded` and `physical_exclusivity=false`.
Correctness that passes this route is device evidence for the exact workload;
latency comparisons require separate paired, interference-aware measurement.
Never turn a CPU smoke, partial workload, or local serialization into full
suite correctness or a performance claim.

Older 2026-10-01 receipts under `docs/DEVICE-VALIDATION-2026-10-01.md` came
from a different admission implementation. They remain historical evidence
under their original IDs and are not silently reclassified as runs of this
entry. Run a fresh smoke and then the complete original workload set through
this gateway before promoting its device status.
