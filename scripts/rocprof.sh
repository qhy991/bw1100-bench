#!/usr/bin/env bash
# Run a diagnostic rocprof pass through this repository's HCU admission.
set -euo pipefail
(( $# >= 6 )) || {
  printf 'usage: rocprof.sh IMAGE RECEIPT PMC_FILE REPORT_CSV KERNEL_REGEX COMMAND [ARGS...]\n' >&2
  exit 2
}
image=$1
receipt=$2
pmc=$3
report=$4
kernel_regex=$5
shift 5
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
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
