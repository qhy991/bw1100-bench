#!/usr/bin/env bash
set -euo pipefail
mode=${1:?}; shift
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P);cd "$root"
export BWBENCH_QUEUE_TIMEOUT=${BWBENCH_QUEUE_TIMEOUT:-300}
export BWBENCH_LATEST_START_EPOCH=$(python3 - <<'PY'
import json,os,time
from pathlib import Path
d=json.loads(Path('campaign/deadline.json').read_text());p=json.loads(Path('campaign/protocol.json').read_text());timeout=int(os.environ.get('BWBENCH_TIMEOUT','600'))
if not 10<=timeout<=900:raise SystemExit('GPU execution bound must stay10..900seconds')
latest=min(d['search_stop_at_epoch']-timeout-30,d['stop_at_epoch']-timeout-30)
if time.time()>=latest:raise SystemExit('No new search GPU admission fits the declared budget')
print(latest)
PY
)
case "$mode" in
 gpu) image=$1;receipt=$2;shift 2;exec bash scripts/dtk.sh gpu "$image" "$receipt" env TRITON_F32_DEFAULT=ieee FLA_TRIL_PRECISION=ieee "$@";;
 profile) receipt=$1;pmc=$2;report=$3;regex=$4;shift 4;exec bash scripts/rocprof.sh "$receipt" "$pmc" "$report" "$regex" env TRITON_F32_DEFAULT=ieee FLA_TRIL_PRECISION=ieee "$@";;
 *) exit 2;;
esac
