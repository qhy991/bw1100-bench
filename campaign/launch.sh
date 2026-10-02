#!/usr/bin/env bash
set -euo pipefail
group=${1:?usage: launch.sh a|b|c|d|e|f}
[[ $group =~ ^[a-f]$ ]]
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$root"
[[ $(hostname) == node4 ]]
bwbench_user_home=/data3/testuser01
export HOME="$bwbench_user_home" PATH="$bwbench_user_home/.local/bin:$PATH"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
flow_path="$bwbench_user_home/.humanize/flowverses/official/flows/ralph_loop/__init__.py"
[[ $(sha256sum "$flow_path" | cut -d ' ' -f1) == dff76869f6c823cb6136e0647d4a52777d1580bb05eff36806211182c946a852 ]]
image=sha256:3ad0ae7192b8f9bafdf5b48fc414f8785f3c2463005e6b25290b7f75146ff260
[[ $(docker image inspect "$image" --format '{{.Id}}') == "$image" ]]
test -r "$bwbench_user_home/.agents/skills/rocm-kernelwiki/SKILL.md"
export HIP_VISIBLE_DEVICES=$(python3 -c 'import json,sys; print(json.load(open("campaign/groups/"+sys.argv[1]+".json"))["hcu"])' "$group")
python3 - <<'PY'
import os
from scripts.hcu_run import _vram, _kfd_visible
if _vram(int(os.environ['HIP_VISIBLE_DEVICES'])) != '0%' or _kfd_visible():
    raise SystemExit('Assigned HCU is not idle by bench admission observations')
PY
profile_instruction=$(python3 scripts/ralph_profile_intake.py campaign/profile-intake.json)
mkdir -p campaign/logs campaign/results campaign/candidates campaign/tools .local/aiter-jit-cache .local/triton-cache
bash scripts/dtk.sh cpu "$image" /usr/bin/timeout -k 10s 180s python3 bwbench.py audit --output campaign/results/intake-audit.json
python3 campaign/prepare.py "$group"
test ! -e campaign/logs/ralph.log
test ! -e campaign/exit-code
set +e
hmz exec -f campaign/ralph_flow.py -a claude/glm-5.3:high -c campaign/budget-3h.yaml \
  "$profile_instruction"$'\n\n'"$(cat campaign/TASK.md)" > campaign/logs/ralph.log 2>&1
code=$?
set -e
printf '%s\n' "$code" > campaign/exit-code
exit "$code"

