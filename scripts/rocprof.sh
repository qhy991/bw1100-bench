#!/usr/bin/env bash
# Run a diagnostic rocprof pass through this repository's HCU admission.
# The image comes from the Run's own binding (scripts/pinned_image.py), never from
# the caller: a retyped digest (R6) and a pasted `IMAGE` placeholder (R9) each made
# the gateway write the HCU-wide stop file before any device work.
set -euo pipefail
(( $# >= 5 )) || {
  printf 'usage: rocprof.sh RECEIPT PMC_FILE REPORT_CSV KERNEL_REGEX COMMAND [ARGS...]\n' >&2
  exit 2
}
case "$1" in
  sha256:*|IMAGE)
    printf 'rocprof.sh no longer takes an image; it reads the pinned image from the Run binding.\n' >&2
    printf 'usage: rocprof.sh RECEIPT PMC_FILE REPORT_CSV KERNEL_REGEX COMMAND [ARGS...]\n' >&2
    exit 2 ;;
esac
receipt=$1
pmc=$2
report=$3
kernel_regex=$4
shift 4
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
image=$(python3 "$root/scripts/pinned_image.py" "$root")
# The binding's physical card, so a renumbered index is refused before admission.
expected_pci=$(python3 "$root/scripts/pinned_image.py" --pci "$root")
export BWBENCH_EXPECTED_PCI=$expected_pci
for path in "$pmc" "$report"; do
  case "$path" in
    /*|..|../*|*/../*|*/..) printf 'path must stay relative to the repository: %s\n' "$path" >&2; exit 2 ;;
  esac
done
test -f "$root/$pmc" || { printf 'missing pmc input: %s\n' "$pmc" >&2; exit 2; }
test ! -e "$root/$report" || { printf 'report already exists: %s\n' "$report" >&2; exit 2; }
python3 - "$root" "$root/$pmc" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from scripts.verify_rocprof_csv import requested_metrics
requested_metrics(Path(sys.argv[2]))
PY
mkdir -p "$root/$(dirname "$report")"

bash "$root/scripts/dtk.sh" gpu "$image" "$receipt" \
  /opt/dtk/rocprofiler/bin/rocprof \
  -i "/work/$pmc" -o "/work/$report" "$@"

python3 "$root/scripts/verify_rocprof_csv.py" "$root/$pmc" "$root/$report" "$kernel_regex"
