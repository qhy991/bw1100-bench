#!/usr/bin/env bash
# Self-contained transient DTK container entry for bw1100-bench.
set -euo pipefail
mode=${1:?usage: dtk.sh cpu|gpu IMAGE [RECEIPT] COMMAND [ARGS...]}
image=${2:?image is required}
shift 2
(( $# > 0 )) || { printf 'A command is required\n' >&2; exit 2; }
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
case "$mode" in
  cpu)
    args=(--rm --network none --read-only --tmpfs /tmp:rw,exec,mode=1777,size=2g
          -v "$root:/work" -w /work -v /opt/hyhal:/opt/hyhal:ro
          -e PYTHONDONTWRITEBYTECODE=1 -e HOME=/tmp
          -e "USER=$(id -un)" -e "LOGNAME=$(id -un)" --runtime runc
          --user "$(id -u):$(id -g)"
          -e HIP_VISIBLE_DEVICES=-1 -e ROCR_VISIBLE_DEVICES=-1 -e CUDA_VISIBLE_DEVICES=-1)
    exec docker run "${args[@]}" --entrypoint bash "$image" -c \
      'source /opt/dtk/env.sh; export LD_LIBRARY_PATH="/opt/hyhal/lib:${LD_LIBRARY_PATH:-}"; export PYTHONPATH=/work/.deps/sol-execbench/src; exec "$@"' bash "$@"
    ;;
  gpu)
    : "${HIP_VISIBLE_DEVICES:?pass the observed idle HCU id}"
    receipt=${1:?usage: dtk.sh gpu IMAGE RECEIPT COMMAND [ARGS...]}
    shift
    (( $# > 0 )) || { printf 'A GPU command is required\n' >&2; exit 2; }
    exec python3 "$root/scripts/hcu_run.py" --image "$image" \
      --device "$HIP_VISIBLE_DEVICES" --receipt "$receipt" \
      --timeout "${BWBENCH_TIMEOUT:-180}" -- "$@"
    ;;
  *) printf 'Unknown mode: %s\n' "$mode" >&2; exit 2 ;;
esac
