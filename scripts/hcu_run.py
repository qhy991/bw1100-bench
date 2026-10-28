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
# Where a receipt may be written: the only places both release checks
# (development_control.released and bench_queue.release_audit) read it, so a gateway
# killed before its terminal can never be certified released (#44). The checks live in the
# Run's adapter (development branch) and keep their copy in scripts/json_stream.py; that
# branch's tests/test_json_stream_release.py holds the two equal.
RECEIPT_ROOTS = ('campaign', '.local')
RECEIPT_UNWALKED_DIRS = frozenset(('.git', '.deps', '.data', 'triton-cache', 'aiter-jit-cache',
                                   '__pycache__', 'notes'))


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
               '0 active drivers', 'no valid dcus',
               'cannot connect to the docker daemon', 'docker.sock: connect: permission denied',
               'hcu.sock: connect: permission denied', 'hsa_status_error_out_of_resources',
               'hcu_device_identity_failed')
    return next((marker for marker in markers if marker in text), None)


# The gateway exposes the leased card as the runtime's own logical device. A command that
# sets one of these hides that card; it is the caller's error, refused before admission so
# it costs that request and not the HCU (r3 swiglu, STOP-hcu2, 2026-10-26).
_VISIBILITY = re.compile(r'(?:^|[;&|(`]|\bexport\s+|\benv\s+(?:-\S+\s+)*|\bdeclare\s+-x\s+)\s*'
                         r'(HIP_VISIBLE_DEVICES|ROCR_VISIBLE_DEVICES|CUDA_VISIBLE_DEVICES|'
                         r'DCU_VISIBLE_DEVICES|GPU_DEVICE_ORDINAL)\+?=')


def _visibility_override(command):
    """The device-visibility variable a command assigns inside the container, if any.

    It reads assignments that are a command prefix, an env/export/declare -x argument, or an
    argv word of its own; a mention in an echo or grep is not an assignment. A program that
    sets the variable from inside a script is not visible in argv and keeps today's
    runtime-marker handling.
    """
    for index, word in enumerate(command):
        # An argv word is an assignment only as the first word or right after env/export.
        if re.fullmatch(r'(HIP|ROCR|CUDA|DCU)_VISIBLE_DEVICES\+?=.*|GPU_DEVICE_ORDINAL\+?=.*', word) and (
                index == 0 or command[index - 1] in ('env', 'export')):
            return word.split('=')[0].rstrip('+')
        # A shell string (bash -c '...') can assign anywhere a command starts.
        found = _VISIBILITY.search(word) if ' ' in word else None
        if found:
            return found.group(1)
    return None


def _holders(device):
    """KFD holders of `device` from hy-smi --showpids: every PID whose HCU Index names it."""
    output = subprocess.check_output(['hy-smi', '--showpids'], universal_newlines=True, timeout=15)
    holders = []
    for block in re.split(r'(?m)^\s*(?=PID:)', output):
        pid = re.search(r'PID:\s*(\d+)', block)
        if not pid:
            continue
        index = re.search(r'HCU Index:\s*\[([^\]]*)\]', block)
        indices = re.findall(r"'(\d+)'", index.group(1)) if index else []
        if not indices:
            # A holder whose card cannot be read could be on this card: report it as unknown.
            return None
        if str(device) in indices:
            holders.append(int(pid.group(1)))
    return holders


def _started_after(pid, epoch, proc=Path('/proc')):
    """Whether a live, non-zombie process started after `epoch` (stat field 22 + btime)."""
    try:
        stat = (proc / str(pid) / 'stat').read_text()
        boot = next(int(line.split()[1]) for line in (proc / 'stat').read_text().splitlines()
                    if line.startswith('btime '))
    except (OSError, StopIteration, ValueError):
        return False
    try:
        fields = stat[stat.rindex(')') + 2:].split()
        if fields[0] == 'Z':
            return False
        return boot + int(fields[19]) / os.sysconf('SC_CLK_TCK') > epoch
    except (ValueError, IndexError):
        return False


def _select_runtime_device(device, bus_output, cdi, drm_root=Path('/sys/class/drm')):
    """Resolve the runtime's CDI selector by PCI, not dcu-ctk list ordinal."""
    buses = re.findall(r'HCU\[(\d+)\].*?PCI Bus:\s*([0-9a-fA-F:.]+)', bus_output)
    matches = [pci.lower() for index, pci in buses if int(index) == device]
    if len(matches) != 1:
        raise RuntimeError('selected HCU has no unique physical PCI binding')
    if cdi.get('kind') != 'c-3000.com/hcu':
        raise RuntimeError('unexpected installed DCU CDI device kind')
    candidates = []
    for record in cdi['devices']:
        selector = str(record['name'])
        if not selector.isdigit():
            continue
        nodes = [n['path'] for n in record.get('containerEdits', {}).get('deviceNodes', [])
                 if re.fullmatch(r'/dev/dri/renderD\d+', n['path'])]
        if len(nodes) != 1:
            raise RuntimeError('single-card CDI selector must have one render device')
        pci = (drm_root / Path(nodes[0]).name / 'device').resolve().name.lower()
        if pci == matches[0]:
            candidates.append({'runtime': 'dcu', 'device_selector': selector,
                               'pci_bus_id': pci, 'cdi_device': cdi['kind'] + '=' + selector,
                               'render_node': nodes[0]})
    if len(candidates) != 1:
        raise RuntimeError('DCU runtime inventory does not uniquely match selected physical HCU')
    return candidates[0]


def _runtime_binding(device):
    bus = subprocess.check_output(['hy-smi', '--showbus'], universal_newlines=True, timeout=15)
    cdi = json.loads(subprocess.check_output(
        ['dcu-ctk', 'cdi', 'generate', '--format', 'json'], universal_newlines=True, timeout=30))
    return _select_runtime_device(device, bus, cdi)


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


def _scope_parts(path):
    """The resolved receipt path's components below ROOT, refused outside the receipt scope."""
    root = ROOT.resolve()
    if os.path.commonpath((str(root), str(path))) != str(root):
        raise ValueError('receipt must stay inside the repository')
    parts = path.relative_to(root).parts
    if not parts or parts[0] not in RECEIPT_ROOTS or path.suffix != '.json':
        raise ValueError('receipt must be a .json file under campaign/ or .local/, where the '
                         'release checks read it: %s' % '/'.join(parts))
    unwalked = sorted(RECEIPT_UNWALKED_DIRS.intersection(parts[:-1]))
    if unwalked:
        raise ValueError('receipt must not be under %s, which the release checks skip: %s'
                         % (', '.join(unwalked), '/'.join(parts)))
    return root, parts


def _write_receipt(path, value):
    """Create a receipt, re-checking its scope at the moment of creation (#45 review).

    The lock wait can last minutes. A directory swapped for a symlink in that time must not
    move the receipt where the release checks do not read it, inside or outside the
    repository. Every directory below ROOT is opened without following a symlink and the
    file is created exclusively relative to the last one, so a swapped component fails the
    open instead of redirecting it.
    """
    root, parts = _scope_parts(path)
    directory = os.open(str(root), os.O_RDONLY | os.O_DIRECTORY)
    try:
        for index, part in enumerate(parts[:-1]):
            try:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            except OSError as error:
                raise ValueError('receipt directory %s is no longer a plain directory: %s'
                                 % ('/'.join(parts[:index + 1]), error.strerror)) from error
            os.close(directory)
            directory = child
        created = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                          0o600, dir_fd=directory)
    finally:
        os.close(directory)
    os.fchmod(created, 0o600)
    with os.fdopen(created, 'w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def _receipt_paths(name):
    path = (ROOT.resolve() / name).resolve()
    try:
        _scope_parts(path)
        if path.name.endswith('-terminal.json'):
            # The checks read such a file as a terminal and never look for its own terminal.
            raise ValueError('receipt must not end in -terminal.json, which the release checks '
                             'read as a terminal: %s' % path.name)
    except ValueError as error:
        raise ValueError('%s (from --receipt %s)' % (error, name)) from None
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


def _foreign_occupancy(device, exited_epoch):
    """The holders of a card that stayed occupied after this job's container exited, when
    every one of them started after that exit and so cannot be this job's.

    This decides only whether an unconfirmed release is attributable to someone else (r3
    STOP-hcu3, 2026-10-26). It never makes a release confirmed: the receipt keeps the
    observed VRAM and the not-qualified status. A lingering holder of ours started before
    the exit and keeps the STOP; an empty or unreadable holder list keeps it too.
    """
    holders = _holders(device)
    if holders and all(_started_after(pid, exited_epoch) for pid in holders):
        return holders
    return None


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
    parser.add_argument('--expected-pci',
                        help='refuse unless --device resolves to this physical PCI bus id; hy-smi '
                             'indices shift when a card leaves the bus (node4, 2026-10-28)')
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
    if args.expected_pci is not None and not re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]',
                                                           args.expected_pci.lower()):
        parser.error('expected PCI must be domain:bus:device.function')
    override = _visibility_override(command)
    if override:
        print(json.dumps({'status': 'refused_before_admission', 'reason': 'device_visibility_override',
                          'variable': override,
                          'repair': 'inside the admitted container keep the gateway-established logical '
                                    'device; set ' + override + ' only for a host-side wrapper'}), flush=True)
        return 64
    _check_stop()
    try:
        receipt, terminal = _receipt_paths(args.receipt)
    except ValueError as error:
        parser.error(str(error))
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = LOCK_DIR / ('hcu-%d.lock' % args.device)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        queue_wait_seconds = _acquire_hcu_lock(fd, args.queue_timeout, args.latest_start_epoch)
        try:
            binding = _runtime_binding(args.device)
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError, KeyError) as error:
            _stop_for_unknown_release(None, 'physical_device_binding_failed: ' + str(error))
            raise
        # Before observing the card: after a renumbering the index may name another user's
        # busy card, and that must read as a renumbering, not as a held card.
        if args.expected_pci is not None and binding['pci_bus_id'] != args.expected_pci.lower():
            # A renumbered card, not a held one: nothing was admitted. The ids go on their own
            # line so the last line stays the fixed refusal the adapter matches.
            print(json.dumps({'status': 'refused_before_admission', 'reason': 'hcu_pci_mismatch',
                              'device': args.device, 'observed_pci': binding['pci_bus_id'],
                              'expected_pci': args.expected_pci.lower()}), flush=True)
            raise RuntimeError('selected HCU is not the expected physical PCI device; no GPU admitted')
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
            'expected_pci': args.expected_pci.lower() if args.expected_pci else None,
            'started_at': started,
        }
        _write_receipt(receipt, admission)
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
            '-e', 'DCU_VISIBLE_DEVICES=' + binding['device_selector'],
            '--runtime', 'dcu',
            '--entrypoint', 'bash', image_id,
            '-c', bootstrap,
            'bash', str(child_timeout), 'bash', '-c',
            'python3 /work/scripts/hcu_device_identity.py "$1" || exit $?; shift; exec "$@"',
            'bash', binding['pci_bus_id'],
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
        exited_epoch = time.time()
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
        foreign = None
        if (container_live is False and after_vram not in ('0%', None) and exit_code == 0
                and not after_kfd):
            try:
                foreign = _foreign_occupancy(args.device, exited_epoch)
            except (OSError, subprocess.SubprocessError, ValueError):
                foreign = None
        if foreign:
            # Still not_qualified with its observed VRAM; only the HCU-wide STOP is withheld.
            terminal_doc['foreign_occupancy'] = {'holders': foreign, 'rule': 'every holder started after '
                                                 'this container exited'}
        _write_receipt(terminal, terminal_doc)
        if not foreign and (after_vram != '0%' or after_kfd is not False or container_live is not False):
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
