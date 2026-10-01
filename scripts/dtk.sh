#!/usr/bin/env bash
# Transient-container entry, following Cake's documented DCU execution pattern.
set -euo pipefail
mode="${1:?usage: dtk.sh cpu|gpu IMAGE [RECEIPT] COMMAND [ARGS...]}"
image="${2:?image is required}"
shift 2
(( $# > 0 )) || { printf 'A command is required\n' >&2; exit 2; }
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
args=(--rm --network none --read-only --tmpfs /tmp:rw,exec,mode=1777,size=2g
      -v "$root:/work" -w /work
      -v /opt/hyhal:/opt/hyhal:ro -e PYTHONDONTWRITEBYTECODE=1)
case "$mode" in
  cpu)
    args+=(--runtime runc --user "$(id -u):$(id -g)"
           -e HIP_VISIBLE_DEVICES=-1 -e ROCR_VISIBLE_DEVICES=-1 -e CUDA_VISIBLE_DEVICES=-1)
    command=("$@")
    ;;
  gpu)
    : "${HIP_VISIBLE_DEVICES:?pass the HIP device assigned to this experiment}"
    : "${CAKE_CHECKOUT:?pass the inspected Open-Cake checkout for Hygon admission}"
    [[ -f "$CAKE_CHECKOUT/src/open_cake_ir/evaluation/local_broker.py" ]] || {
      printf 'CAKE_CHECKOUT has no Hygon local_broker.py\n' >&2; exit 2;
    }
    receipt="${1:?usage: dtk.sh gpu IMAGE RECEIPT COMMAND [ARGS...]}"
    shift
    (( $# > 0 )) || { printf 'A GPU command is required\n' >&2; exit 2; }
    case "$receipt" in
      /*|..|../*|*/..|*/../*) printf 'Receipt must stay inside the project\n' >&2; exit 2 ;;
    esac
    mkdir -p "$root/$(dirname "$receipt")"
    [[ ! -e "$root/$receipt" ]] || { printf 'Receipt already exists\n' >&2; exit 2; }
    args+=(--runtime dtk --privileged --device /dev/kfd --device /dev/dri
           --device /dev/mkfd -v "$CAKE_CHECKOUT:/cake:ro" -e HIP_VISIBLE_DEVICES
           -e PYTHONPATH=/cake/src:/work/.deps/sol-execbench/src)
    command=(python3 /work/scripts/run_with_cake_local_broker.py
             --receipt "$receipt" --timeout 180 -- "$@")
    ;;
  *) printf 'Unknown mode: %s\n' "$mode" >&2; exit 2 ;;
esac
exec docker run "${args[@]}" --entrypoint bash "$image" -c \
  'source /opt/dtk/env.sh; export LD_LIBRARY_PATH="/opt/hyhal/lib:${LD_LIBRARY_PATH:-}"; export PYTHONPATH=/work/.deps/sol-execbench/src${PYTHONPATH:+:$PYTHONPATH}; exec "$@"' bash "${command[@]}"
