"""Bounded Hygon HCU correctness run with task-owned local serialization."""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
LOCK_DIR = Path.home() / '.local/state/bw1100-bench'


def _check_stop():
    name = os.environ.get('BWBENCH_STOP_FILE')
    if name and Path(name).exists():
        raise RuntimeError('experiment owner stopped new GPU admissions')


def _stop_for_unknown_release(job_id, reason):
    name = os.environ.get('BWBENCH_STOP_FILE')
    if name:
        try:
            _write_new(Path(name), {'job_id': job_id, 'reason': reason,
                                    'node_epoch': time.time()})
        except FileExistsError:
            pass


def _runtime_error_marker(output):
    text = (output or '').lower()
    markers = ('no cuda gpus are available', 'no hip gpus are available',
               '0 active drivers', 'no valid dcus', 'dcus as they are invalid',
               'cannot connect to the docker daemon', 'docker.sock: connect: permission denied',
               'hcu.sock: connect: permission denied', 'hsa_status_error_out_of_resources')
    return next((marker for marker in markers if marker in text), None)


def _select_runtime_device(device, bus_output, inventory, drm_root=Path('/sys/class/drm')):
    """Join the two installed inventories by PCI, never by their ordinal."""
    buses = re.findall(r'HCU\[(\d+)\].*?PCI Bus:\s*([0-9a-fA-F:.]+)', bus_output)
    matches = [pci.lower() for index, pci in buses if int(index) == device]
    if len(matches) != 1:
        raise RuntimeError('selected HCU has no unique physical PCI binding')
    candidates = []
    for index, unique_id, render in re.findall(
            r'^\s*(\d+)\s+(0x[0-9a-fA-F]+)\s+/dev/dri/(renderD\d+)\s*$',
            inventory, re.MULTILINE):
        pci = (drm_root / render / 'device').resolve().name.lower()
        if pci == matches[0]:
            candidates.append({'runtime': 'dcu', 'dcu_index': int(index),
                               'pci_bus_id': pci, 'dcu_unique_id': unique_id,
                               'render_node': '/dev/dri/' + render})
    if len(candidates) != 1:
        raise RuntimeError('DCU runtime inventory does not uniquely match selected physical HCU')
    return candidates[0]


def _runtime_binding(device):
    bus = subprocess.check_output(['hy-smi', '--showbus'], universal_newlines=True, timeout=15)
    inventory = subprocess.check_output(['dcu-ctk', 'dcu', 'list'], universal_newlines=True, timeout=15)
    return _select_runtime_device(device, bus, inventory)


def _check_latest_start(latest_start_epoch):
    if latest_start_epoch is not None and time.time() >= latest_start_epoch:
        raise RuntimeError('GPU admission window expired before device execution')


def _acquire_hcu_lock(fd, queue_timeout, latest_start_epoch=None):
    """Bounded waiting on the existing per-user lock; no second allocator.

    This serializes only callers that use the same lock. It is not FIFO or
    physical exclusivity against other users. The default preserves fail-fast.
    """
    started = time.monotonic() if queue_timeout else None
    while True:
        _check_stop()
        _check_latest_start(latest_start_epoch)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if not queue_timeout:
                raise RuntimeError('another local bw1100-bench job holds this HCU lock')
            remaining = queue_timeout - (time.monotonic() - started)
            if latest_start_epoch is not None:
                remaining = min(remaining, latest_start_epoch - time.time())
            if remaining <= 0:
                raise RuntimeError('timed out waiting for this HCU lock; no GPU admitted')
            time.sleep(min(0.1, remaining))
        else:
            _check_stop()
            return time.monotonic() - started if started is not None else 0.0


def _vram(device):
    output = subprocess.check_output(['hy-smi'], universal_newlines=True, timeout=15)
    rows = [line.split() for line in output.splitlines()
            if line.split() and line.split()[0] == str(device)]
    if len(rows) != 1:
        raise RuntimeError('selected HCU is absent from hy-smi')
    return rows[0][5]


def _kfd_visible():
    return subprocess.run(['fuser', '/dev/kfd'], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, timeout=10).returncode == 0


def _write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    os.chmod(str(path), 0o600)


def _receipt_paths(name):
    root = ROOT.resolve()
    path = (root / name).resolve()
    if os.path.commonpath((str(root), str(path))) != str(root):
        raise ValueError('receipt must stay inside the repository')
    terminal = path.with_name(path.stem + '-terminal.json')
    if path.exists() or terminal.exists():
        raise FileExistsError('receipt path already exists')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path, terminal


def _observe_release(device, job_id, exit_code):
    started = time.monotonic()
    observations = []
    while True:
        vram, kfd = _vram(device), _kfd_visible()
        live = bool(subprocess.check_output(
            ['docker', 'ps', '-q', '--filter', 'name=^/' + job_id + '$'],
            universal_newlines=True, timeout=15).strip())
        elapsed = time.monotonic() - started
        observations.append({'elapsed_s': elapsed, 'vram': vram,
                             'kfd_visible': kfd, 'container_still_running': live})
        if (vram == '0%' and not kfd and not live) or exit_code != 0 or live or elapsed >= 30:
            return vram, kfd, live, observations
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--device', required=True, type=int)
    parser.add_argument('--receipt', required=True)
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--queue-timeout', type=int, default=0,
                        help='wait up to this many seconds for the existing HCU lock (0 fails fast)')
    parser.add_argument('--latest-start-epoch', type=float,
                        help='refuse device execution at/after this task-owned node deadline')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == '--' else args.command
    if not command or args.device < 0 or not 10 <= args.timeout <= 3600:
        parser.error('command, nonnegative HCU id and timeout 10..3600 required')
    if not 0 <= args.queue_timeout <= 3600:
        parser.error('queue timeout must be in 0..3600 seconds')
    if args.latest_start_epoch is not None and (
            not math.isfinite(args.latest_start_epoch) or args.latest_start_epoch <= 0):
        parser.error('latest start epoch must be finite and positive')
    _check_stop()
    receipt, terminal = _receipt_paths(args.receipt)
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = LOCK_DIR / ('hcu-%d.lock' % args.device)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        queue_wait_seconds = _acquire_hcu_lock(fd, args.queue_timeout, args.latest_start_epoch)
        try:
            image_id = subprocess.check_output(
                ['docker', 'image', 'inspect', args.image, '--format', '{{.Id}}'],
                universal_newlines=True, timeout=15).strip()
            before_vram = _vram(args.device)
            before_kfd = _kfd_visible()
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            _stop_for_unknown_release(None, 'pre_admission_observation_failed: ' + str(error))
            raise
        if before_vram != '0%' or before_kfd:
            raise RuntimeError('selected HCU is not idle by observed VRAM/KFD check')
        # Device/image probes may consume the remaining admission window too.
        _check_latest_start(args.latest_start_epoch)
        try:
            binding = _runtime_binding(args.device)
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            _stop_for_unknown_release(None, 'physical_device_binding_failed: ' + str(error))
            raise
        _check_latest_start(args.latest_start_epoch)
        _check_stop()
        job_id = 'bw-' + uuid.uuid4().hex[:12]
        started = time.time()
        reserve = min(30, max(2, args.timeout // 3))
        child_timeout = args.timeout - reserve
        kill_after = max(1, reserve // 2)
        admission = {
            'schema': 'bw1100-bench.hcu-admission.v1',
            'job_id': job_id,
            'mode': 'local_serialized',
            'external_gpu_activity': 'not_excluded',
            'physical_exclusivity': False,
            'selected_hcu': args.device,
            'runtime_device_binding': binding,
            'image_id': image_id,
            'before_vram': before_vram,
            'before_kfd_visible': before_kfd,
            'command': command,
            'timeout_s': args.timeout,
            'container_timeout_s': child_timeout,
            'queue_timeout_s': args.queue_timeout,
            'queue_wait_s': queue_wait_seconds,
            'latest_start_epoch': args.latest_start_epoch,
            'started_at': started,
        }
        _write_new(receipt, admission)
        bootstrap = (
            'source /opt/dtk/env.sh; '
            'export LD_LIBRARY_PATH="/opt/hyhal/lib:${LD_LIBRARY_PATH:-}"; '
            'export PYTHONPATH=/work/.deps/sol-execbench/src; '
            'exec /usr/bin/timeout --signal=TERM --kill-after=%ds "$@"'
        ) % kill_after
        docker_command = [
            'docker', 'run', '--rm', '--name', job_id,
            '--network', 'none', '--read-only',
            '--tmpfs', '/tmp:rw,exec,mode=1777,size=2g',
            '-v', str(ROOT) + ':/work', '-v', '/opt/hyhal:/opt/hyhal:ro',
            '-w', '/work', '-e', 'PYTHONDONTWRITEBYTECODE=1',
            '-e', 'HOME=/tmp',
            # The runtime exposes only this PCI-bound card and HIP renumbers it to 0.
            # Reapplying the host HCU ordinal as HIP_VISIBLE_DEVICES hides the card.
            '-e', 'DCU_VISIBLE_DEVICES=%d' % binding['dcu_index'],
            '--runtime', 'dcu',
            '--entrypoint', 'bash', image_id,
            '-c', bootstrap,
            'bash', str(child_timeout),
        ] + command
        output = ''
        try:
            child = subprocess.run(docker_command, timeout=args.timeout, check=False,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   universal_newlines=True)
            output = child.stdout or ''
            exit_code = child.returncode
            reason = ('normal_exit' if exit_code == 0 else
                      'container_timeout_or_exit_124' if exit_code == 124 else
                      'nonzero_exit')
        except subprocess.TimeoutExpired as error:
            output = error.output or ''
            if isinstance(output, bytes):
                output = output.decode(errors='replace')
            exit_code, reason = 124, 'timeout_check_container_state'
        except OSError as error:
            exit_code, reason = 127, 'container_launch_error: ' + str(error)
        if output:
            print(output, end='' if output.endswith('\n') else '\n', flush=True)
        runtime_error = _runtime_error_marker(output) if exit_code else None
        time.sleep(1)
        try:
            after_vram, after_kfd, container_live, release_observations = _observe_release(
                args.device, job_id, exit_code)
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            after_vram, after_kfd, container_live = None, None, None
            reason = 'release_observation_failed'
            release_observations = [{'error': type(error).__name__, 'detail': str(error)[:500]}]
        terminal_doc = dict(admission)
        terminal_doc.update(
            exit_code=exit_code, reason=reason,
            runtime_error_marker=runtime_error,
            after_vram=after_vram, after_kfd_visible=after_kfd,
            container_still_running=container_live,
            release_observations=release_observations,
            completed_at=time.time(),
            status=('completed' if exit_code == 0 and after_vram == '0%'
                    and after_kfd is False and container_live is False else 'not_qualified'),
        )
        _write_new(terminal, terminal_doc)
        if after_vram != '0%' or after_kfd is not False or container_live is not False:
            _stop_for_unknown_release(job_id, 'device_release_not_confirmed: ' + reason)
        elif runtime_error:
            _stop_for_unknown_release(job_id, 'runtime_execution_failed: ' + runtime_error)
        print(json.dumps({'job_id': job_id, 'status': terminal_doc['status'],
                          'exit_code': exit_code}), flush=True)
        return 0 if terminal_doc['status'] == 'completed' else exit_code or 75
    finally:
        os.close(fd)


if __name__ == '__main__':
    sys.exit(main())
