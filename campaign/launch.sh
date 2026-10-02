#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$root"
[[ $(hostname) == node4 ]] || { printf 'Launch only on node4\n' >&2; exit 2; }
bwbench_user_home=/data3/testuser01
export PATH="$bwbench_user_home/.local/bin:$PATH"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
flow_path="$bwbench_user_home/.humanize/flowverses/official/flows/ralph_loop/__init__.py"
flow_hash=$(sha256sum "$flow_path" | cut -d ' ' -f1)
[[ $flow_hash == dff76869f6c823cb6136e0647d4a52777d1580bb05eff36806211182c946a852 ]] || {
  printf 'Installed official Ralph flow differs: %s\n' "$flow_hash" >&2; exit 4;
}
image=sha256:3ad0ae7192b8f9bafdf5b48fc414f8785f3c2463005e6b25290b7f75146ff260
[[ $(docker image inspect "$image" --format '{{.Id}}') == "$image" ]]
test -r "$bwbench_user_home/.agents/skills/rocm-kernelwiki/SKILL.md"
env HOME="$bwbench_user_home" python3 - <<'PY'
from scripts.hcu_run import _vram, _kfd_visible
if _vram(2) != '0%' or _kfd_visible():
    raise SystemExit('HCU2 is not idle by the bench admission observations')
PY
profile_instruction=$(env HOME="$bwbench_user_home" python3 \
  scripts/ralph_profile_intake.py campaign/profile-intake.json)
mkdir -p campaign/logs campaign/results .local/aiter-jit-cache
bash scripts/dtk.sh cpu "$image" python3 bwbench.py audit \
  --output campaign/results/intake-audit.json
env HOME="$bwbench_user_home" python3 - <<'PY'
import json, os, subprocess, time
from pathlib import Path
started = int(time.time())
shared = json.loads(Path('/data3/testuser01/experiments/bw1100-bench-ralph-profiled-3h-20261002/campaign/deadline.json').read_text())
if started + 5400 > shared['stop_at_epoch']:
    raise SystemExit('The ninety-minute phase no longer fits the three-hour window')
with Path('campaign/deadline.json').open('x') as stream:
    json.dump({'started_at_epoch': started, 'stop_at_epoch': started + 5400,
               'budget_hours': 1.5, 'host': os.uname().nodename,
               'flow': 'official ralph_loop', 'model': 'claude/glm-5.3:high',
               'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                                       universal_newlines=True).strip(),
               'selected_hcu': 2}, stream, indent=2)
    stream.write('\n')
PY
test ! -e campaign/logs/ralph.log
test ! -e campaign/exit-code
set +e
env HOME="$bwbench_user_home" hmz exec -f "$flow_path" -a claude/glm-5.3:high \
  -c campaign/budget-90m.yaml "$profile_instruction"$'\n\n'"$(cat campaign/TASK.md)" \
  > campaign/logs/ralph.log 2>&1
code=$?
set -e
printf '%s\n' "$code" > campaign/exit-code
exit "$code"
