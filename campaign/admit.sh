#!/usr/bin/env bash
set -euo pipefail
mode=${1:?usage: admit.sh gpu|profile IMAGE ...}
shift
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$root"
python3 - <<'PY'
import json,os,time
from pathlib import Path
deadline=json.loads(Path('campaign/deadline.json').read_text())['stop_at_epoch']
reserve=int(json.loads(Path('campaign/protocol.json').read_text())['final_handoff_minutes']*60)
timeout=int(os.environ.get('BWBENCH_TIMEOUT','600'))
now=time.time()
if not 10 <= timeout <= 900 or now >= deadline-reserve or now+timeout+30 > deadline:
    raise SystemExit('No new GPU admission: timeout does not fit the remaining search window')
PY
case "$mode" in
  gpu) exec bash scripts/dtk.sh gpu "$@" ;;
  profile) exec bash scripts/rocprof.sh "$@" ;;
  *) printf 'Unknown admission mode\n' >&2; exit 2 ;;
esac
