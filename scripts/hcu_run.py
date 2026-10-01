"""Bounded Hygon HCU correctness run with task-owned local serialization."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
LOCK_DIR = Path.home() / '.local/state/bw1100-bench'


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--device', required=True, type=int)
    parser.add_argument('--receipt', required=True)
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == '--' else args.command
    if not command or args.device < 0 or not 1 <= args.timeout <= 3600:
        parser.error('command, nonnegative HCU id and timeout 1..3600 required')
    receipt, terminal = _receipt_paths(args.receipt)
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = LOCK_DIR / ('hcu-%d.lock' % args.device)
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('another local bw1100-bench job holds this HCU lock')
        image_id = subprocess.check_output(
            ['docker', 'image', 'inspect', args.image, '--format', '{{.Id}}'],
            universal_newlines=True, timeout=15).strip()
        before_vram = _vram(args.device)
        before_kfd = _kfd_visible()
        if before_vram != '0%' or before_kfd:
            raise RuntimeError('selected HCU is not idle by observed VRAM/KFD check')
        job_id = 'bw-' + uuid.uuid4().hex[:12]
        started = time.time()
        admission = {
            'schema': 'bw1100-bench.hcu-admission.v1',
            'job_id': job_id,
            'mode': 'local_serialized',
            'external_gpu_activity': 'not_excluded',
            'physical_exclusivity': False,
            'selected_hcu': args.device,
            'image_id': image_id,
            'before_vram': before_vram,
            'before_kfd_visible': before_kfd,
            'command': command,
            'timeout_s': args.timeout,
            'started_at': started,
        }
        _write_new(receipt, admission)
        docker_command = [
            'docker', 'run', '--rm', '--name', job_id,
            '--network', 'none', '--read-only',
            '--tmpfs', '/tmp:rw,exec,mode=1777,size=2g',
            '-v', str(ROOT) + ':/work', '-v', '/opt/hyhal:/opt/hyhal:ro',
            '-w', '/work', '-e', 'PYTHONDONTWRITEBYTECODE=1',
            '-e', 'HIP_VISIBLE_DEVICES=%d' % args.device,
            '--runtime', 'dtk', '--privileged',
            '--device', '/dev/kfd', '--device', '/dev/dri', '--device', '/dev/mkfd',
            '--entrypoint', 'bash', image_id,
            '-c', 'source /opt/dtk/env.sh; export LD_LIBRARY_PATH="/opt/hyhal/lib:${LD_LIBRARY_PATH:-}"; export PYTHONPATH=/work/.deps/sol-execbench/src; exec "$@"',
            'bash',
        ] + command
        try:
            child = subprocess.run(docker_command, timeout=args.timeout, check=False)
            exit_code, reason = child.returncode, 'normal_exit'
        except subprocess.TimeoutExpired:
            exit_code, reason = 124, 'timeout_check_container_state'
        time.sleep(1)
        after_vram = _vram(args.device)
        after_kfd = _kfd_visible()
        container_live = bool(subprocess.check_output(
            ['docker', 'ps', '-q', '--filter', 'name=^/' + job_id + '$'],
            universal_newlines=True, timeout=15).strip())
        terminal_doc = dict(admission)
        terminal_doc.update(
            exit_code=exit_code, reason=reason,
            after_vram=after_vram, after_kfd_visible=after_kfd,
            container_still_running=container_live,
            completed_at=time.time(),
            status=('completed' if exit_code == 0 and after_vram == '0%'
                    and not after_kfd and not container_live else 'not_qualified'),
        )
        _write_new(terminal, terminal_doc)
        print(json.dumps({'job_id': job_id, 'status': terminal_doc['status'],
                          'exit_code': exit_code}), flush=True)
        return 0 if terminal_doc['status'] == 'completed' else exit_code or 75
    finally:
        os.close(fd)


if __name__ == '__main__':
    sys.exit(main())
