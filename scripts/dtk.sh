#!/usr/bin/env bash
# Transient-container entry, following Cake's documented DCU execution pattern.
set -euo pipefail
mode="${1:?usage: dtk.sh cpu|gpu IMAGE COMMAND [ARGS...]}"
image="${2:?image is required}"
shift 2
(( $# > 0 )) || { printf 'A command is required\n' >&2; exit 2; }
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
args=(--rm --network none --read-only --tmpfs /tmp:rw,exec,mode=1777,size=2g
      --user "$(id -u):$(id -g)" -v "$root:/work" -w /work
      -v /opt/hyhal:/opt/hyhal:ro -e PYTHONDONTWRITEBYTECODE=1)
case "$mode" in
  cpu) args+=(--runtime runc -e HIP_VISIBLE_DEVICES=-1 -e ROCR_VISIBLE_DEVICES=-1 -e CUDA_VISIBLE_DEVICES=-1) ;;
  gpu)
    : "${HIP_VISIBLE_DEVICES:?pass the HIP device assigned to this experiment}"
    args+=(--runtime dtk --device /dev/kfd --device /dev/dri -e HIP_VISIBLE_DEVICES)
    ;;
  *) printf 'Unknown mode: %s\n' "$mode" >&2; exit 2 ;;
esac
exec docker run "${args[@]}" --entrypoint bash "$image" -c \
  'source /opt/dtk/env.sh; export LD_LIBRARY_PATH="/opt/hyhal/lib:${LD_LIBRARY_PATH:-}"; export PYTHONPATH=/work/.deps/sol-execbench/src${PYTHONPATH:+:$PYTHONPATH}; exec "$@"' bash "$@"
