"""Retain a same-Run HIP exploration without creating a Cake evaluation."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCHEMA = 'bw1100-bench.native-probe.v1'
ADMISSION = 'bw1100-bench.hcu-admission.v1'
SOURCE_SUFFIXES = {'.py', '.hip', '.cpp', '.cc', '.c', '.cu', '.h', '.hpp', '.hh'}


class NativeReleaseUnknown(ValueError):
    pass


def read(path):
    return json.loads(path.read_text())


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def own_path(root, name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Use a relative path inside this Run')
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('Native evidence cannot use symlinks: ' + name)
    current.resolve().relative_to(root.resolve())
    return current


def freeze(root, folder, request, skip_reason):
    gap = request['gap']
    for key in ('hypothesis', 'operation', 'limitation', 'supported_transforms',
                'proposed_owner', 'counterexamples'):
        if not isinstance(gap.get(key), str) or not gap[key].strip():
            raise ValueError('Gap dossier needs nonempty ' + key)
    evidence = gap['evidence']
    if not isinstance(evidence, list) or not evidence or len(set(evidence)) != len(evidence):
        raise ValueError('Gap dossier needs distinct own evidence paths')
    if request['target'] != 'gfx938':
        raise ValueError('Preserve the assigned gfx938 target')
    snapshots = []
    (folder / 'gap').mkdir()
    for index, name in enumerate(evidence):
        source = own_path(root, name)
        if not source.relative_to(root).parts[:1] == ('campaign',):
            raise ValueError('Gap evidence must belong to this Run campaign')
        retained = folder / 'gap' / ('%03d.source' % index)
        retained.write_bytes(source.read_bytes())
        snapshots.append(dict(original=name, path=str(retained.relative_to(root)), sha256=digest(retained)))
    manifest = dict(schema=SCHEMA, implementation_kind='native_hip', id=folder.name,
                    target=request['target'], gap=gap, gap_evidence=snapshots,
                    source_files=[], started_at_epoch=time.time())
    if not skip_reason:
        source = own_path(root, request['source_dir'])
        relative = source.relative_to(root)
        if len(relative.parts) < 3 or relative.parts[:2] != ('campaign', 'candidates') or not source.is_dir():
            raise ValueError('Source closure must be a dedicated own campaign/candidates directory')
        flags = request['build_flags']
        if not isinstance(flags, list) or any(not isinstance(flag, str) for flag in flags):
            raise ValueError('Declare the native build flags as a string list')
        entry = Path(request['entry'])
        if entry.is_absolute() or '..' in entry.parts or entry.suffix != '.py':
            raise ValueError('Entry must be a Python file relative to the source directory')
        for path in sorted(source.rglob('*')):
            checked = own_path(root, str(path.relative_to(root)))
            if checked.is_dir():
                continue
            if not checked.is_file() or checked.suffix not in SOURCE_SUFFIXES:
                raise ValueError('Source closure contains a non-source file: ' + str(path))
            retained = folder / 'source' / path.relative_to(source)
            retained.parent.mkdir(parents=True, exist_ok=True)
            retained.write_bytes(checked.read_bytes())
            manifest['source_files'].append(dict(path=str(retained.relative_to(root)), sha256=digest(retained)))
        frozen_entry = folder / 'source' / entry
        if not frozen_entry.is_file():
            raise ValueError('Entry is absent from the retained source closure')
        cache = own_path(root, '.local/hip-probes/' + folder.name)
        cache.mkdir(parents=True, exist_ok=False)
        manifest.update(entry=str(frozen_entry.relative_to(root)), entry_symbol='run',
                        source_directory=str((folder / 'source').relative_to(root)),
                        build_directory=str(cache.relative_to(root)), build_flags=flags)
    return manifest


def verify_files(root, manifest, build=None):
    if manifest['source_files']:
        source = own_path(root, manifest['source_directory'])
        actual = set()
        for path in source.rglob('*'):
            checked = own_path(root, str(path.relative_to(root)))
            if checked.is_file():
                actual.add(str(checked.relative_to(root)))
            elif not checked.is_dir():
                raise ValueError('Frozen native source contains a non-file member')
        if actual != {row['path'] for row in manifest['source_files']}:
            raise ValueError('Frozen native source directory membership changed')
    for row in manifest['source_files'] + manifest['gap_evidence']:
        if digest(own_path(root, row['path'])) != row['sha256']:
            raise ValueError('Retained native source or gap evidence changed: ' + row['path'])
    if build is not None:
        for row in build['artifacts']:
            if digest(own_path(root, row['path'])) != row['sha256']:
                raise ValueError('Retained native build artifact changed: ' + row['path'])


def load_entry(root, manifest):
    verify_files(root, manifest)
    entry = own_path(root, manifest['entry'])
    sys.path.insert(0, str(own_path(root, manifest['source_directory'])))
    sys.path.insert(0, str(entry.parent))
    spec = importlib.util.spec_from_file_location('own_native_probe', entry)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, 'run', None)) or not callable(getattr(module, 'prepare', None)):
        raise ValueError('The retained entry must expose prepare() and run()')
    return module


def prepare_native(root, manifest_path):
    manifest = read(own_path(root, manifest_path))
    if manifest['schema'] != SCHEMA or manifest['entry_symbol'] != 'run':
        raise ValueError('Expected the native probe manifest with fixed run() entry')
    cache = own_path(root, manifest['build_directory'])
    os.environ.update(HIP_PROBE_BUILD_DIR=str(cache), TORCH_EXTENSIONS_DIR=str(cache),
                      HIP_PROBE_TARGET=manifest['target'],
                      HIP_PROBE_BUILD_FLAGS=json.dumps(manifest['build_flags']))
    value = load_entry(root, manifest).prepare()
    if not isinstance(value, dict) or not isinstance(value.get('artifacts'), list) or not value['artifacts']:
        raise ValueError('prepare() must declare its persistent native build artifacts')
    artifacts = []
    for name in value['artifacts']:
        path = own_path(cache, name)
        if not path.is_file():
            raise ValueError('Declared build artifact is absent: ' + name)
        artifacts.append(dict(path=str(path.relative_to(root)), sha256=digest(path)))
    verify_files(root, manifest)
    write_new(own_path(root, manifest_path).with_name('build.json'), dict(
        target=manifest['target'], flags=manifest['build_flags'], artifacts=artifacts,
        metadata=value.get('metadata'), scope='declared artifacts; actual native execution requires source review'))


def binding(root):
    development = (root / 'campaign/development-binding.json').exists()
    if development:
        from campaign.development_binding import reconcile
        return 'development', reconcile(root)
    from campaign.binding import reconcile, git
    bound = reconcile(root)
    intake = read(root / 'campaign/intake.json')
    if intake['source_commit'] != git(root, 'rev-parse', 'HEAD'):
        raise ValueError('Prepared Bench source changed after intake')
    return 'bench', bound


def search_window(root, kind, bound, deadline):
    if read(root / 'campaign/deadline.json') != deadline:
        raise ValueError('Original native probe deadline changed')
    if kind == 'development':
        if Path(bound['batch_stop_file']).exists():
            raise ValueError('Existing batch STOP prevents native exploration')
        if (root / 'campaign/SEARCH_CLOSE.json').exists():
            raise ValueError('SEARCH_CLOSE prevents native exploration')
    stop = deadline['search_stop_at_epoch']
    if (not math.isfinite(stop) or stop != deadline['started_at_epoch'] + 9000
            or deadline['stop_at_epoch'] != deadline['started_at_epoch'] + 10800):
        raise ValueError('Original final confirmation reserve is invalid')
    remaining = int(stop - time.time()) - 30
    if remaining < 10:
        raise ValueError('No native stage fits the original search cutoff')
    return deadline, min(600 if kind == 'development' else 900, remaining)


def release(root, receipt, bound, command, cutoff):
    try:
        before = read(own_path(root, receipt))
        after = read(own_path(root, receipt).with_name(Path(receipt).stem + '-terminal.json'))
        if not isinstance(before, dict) or not isinstance(after, dict):
            raise ValueError('Native admission and terminal must be records')
        if (before.get('schema') != ADMISSION or not before.get('job_id')
                or before.get('selected_hcu') != bound['hcu'] or before.get('image_id') != bound['image']
                or before.get('command') != command or any(after.get(k) != v for k, v in before.items())):
            raise ValueError('Native terminal does not match the exact admitted device command')
        if (after.get('container_still_running') is not False or after.get('after_vram') != '0%'
                or after.get('after_kfd_visible') is not False):
            raise ValueError('Native device release is unknown or incomplete')
        started, completed = before['started_at'], after['completed_at']
        if not (math.isfinite(started) and math.isfinite(completed) and started <= completed):
            raise ValueError('Native terminal timestamps are invalid')
        if type(after.get('exit_code')) is not int:
            raise ValueError('Native terminal has no exact exit code')
    except (KeyError, TypeError, ValueError, OSError) as error:
        raise NativeReleaseUnknown(str(error)) from error
    if completed > cutoff:
        raise ValueError('Native stage did not complete inside the original search cutoff')
    return after


def bench_stages(task, entry, folder):
    gate = folder + '/correctness.json'
    return [
        ('correctness', ['python3', 'bwbench.py', 'check', '--task', task, '--device', 'cuda:0',
                         '--candidate', '/work/' + entry, '--symbol', 'run', '--workloads', 'all',
                         '--rounds', '10', '--output', gate]),
        ('caller', ['python3', 'campaign/caller_guards.py', '--candidate', '/work/' + entry,
                    '--output', folder + '/caller.json']),
        ('latency', ['python3', 'campaign/paired_wall.py', '--candidate', '/work/' + entry,
                     '--gate', gate, '--output', folder + '/latency.json',
                     '--implementation-kind', 'native_hip']),
    ]


def run_probe(root, identity, request_name, skip_reason=''):
    if not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}', identity):
        raise ValueError('Use a unique lowercase probe ID')
    root = root.resolve()
    folder = own_path(root, 'campaign/engineering/' + identity)
    folder.mkdir(parents=True, exist_ok=False)
    relative = str(folder.relative_to(root))
    outcome = dict(id=identity, implementation_kind='native_hip', status='not_executed',
                   phase='search', started_at_epoch=time.time(), admission_receipts=[],
                   qualification='none; semantic, precision and execution-source review pending')
    manifest = None
    build = None
    build_hash = None
    try:
        request_path = own_path(root, request_name)
        if request_path.relative_to(root).parts[:2] != ('campaign', 'candidates'):
            raise ValueError('The request must be an own campaign/candidates file')
        raw = request_path.read_bytes()
        (folder / 'request.source').write_bytes(raw)
        request = json.loads(raw)
        kind, bound = binding(root)
        outcome.update(task=bound['task'], compiler=bound.get('compiler', bound.get('compiler_commit')))
        deadline = read(root / 'campaign/deadline.json')
        manifest = freeze(root, folder, request, skip_reason)
        manifest.update(task=outcome['task'], compiler=outcome['compiler'], contract_kind=kind, deadline=deadline)
        manifest_path = folder / 'manifest.json'
        write_new(manifest_path, manifest)
        manifest_hash = digest(manifest_path)
        outcome['manifest_sha256'] = manifest_hash
        def unchanged():
            if digest(manifest_path) != manifest_hash:
                raise ValueError('Native manifest changed during execution')
            if build_hash is not None and digest(folder / 'build.json') != build_hash:
                raise ValueError('Native build report changed during execution')
            verify_files(root, manifest, build)
            if read(root / 'campaign/deadline.json') != deadline:
                raise ValueError('Original native probe deadline changed')
            current_kind, current = binding(root)
            if current_kind != kind or current != bound:
                raise ValueError('Native Run binding changed')
        if skip_reason:
            outcome.update(status='skipped', reason=skip_reason)
        else:
            deadline, timeout = search_window(root, kind, bound, deadline)
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', HIP_VISIBLE_DEVICES=str(bound['hcu']))
            if kind == 'development':
                env['BWBENCH_STOP_FILE'] = bound['batch_stop_file']
            native_env = ['env', 'HIP_PROBE_BUILD_DIR=/work/' + manifest['build_directory'],
                          'TORCH_EXTENSIONS_DIR=/work/' + manifest['build_directory'],
                          'PYTHONPATH=' + ':'.join('/work/' + name for name in
                              (str(Path(manifest['entry']).parent), manifest['source_directory'], '.deps/sol-execbench/src')),
                          'HIP_PROBE_TARGET=gfx938', 'HIP_PROBE_BUILD_FLAGS=' + json.dumps(manifest['build_flags']),
                          'TRITON_F32_DEFAULT=ieee', 'FLA_TRIL_PRECISION=ieee',
                          'AITER_JIT_DIR=/work/.local/aiter-jit-cache', 'TRITON_CACHE_DIR=/work/.local/triton-cache']
            outcome['stage'] = 'prepare'
            with (folder / 'prepare.log').open('x') as log:
                code = subprocess.call(['bash', 'scripts/dtk.sh', 'cpu', bound['image'],
                    '/usr/bin/timeout', '-k', '10s', str(timeout) + 's', 'python3',
                    'campaign/hip_probe.py', 'prepare', '--manifest', relative + '/manifest.json'],
                    cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            unchanged()
            if code:
                raise ValueError('CPU native preparation failed: exit ' + str(code))
            build = read(folder / 'build.json')
            build_hash = digest(folder / 'build.json')
            outcome['build_report_sha256'] = build_hash
            unchanged()
            stages = bench_stages(bound['task'], manifest['entry'], relative) if kind == 'bench' else [
                ('device', ['python3', 'campaign/development_device.py', '--native-manifest',
                            relative + '/manifest.json', '--output', relative + '/observed.pt'])]
            for stage, command in stages:
                outcome['stage'] = stage
                unchanged()
                deadline, timeout = search_window(root, kind, bound, deadline)
                receipt = relative + '/' + stage + '-admission.json'
                outcome['admission_receipts'].append(receipt)
                device_command = native_env + command
                gateway = ['python3', 'scripts/hcu_run.py', '--image', bound['image'],
                    '--device', str(bound['hcu']), '--timeout', str(timeout), '--queue-timeout', '300',
                    '--latest-start-epoch', str(deadline['search_stop_at_epoch'] - timeout - 30),
                    '--receipt', receipt, '--'] + device_command
                with (folder / (stage + '.log')).open('x') as log:
                    code = subprocess.call(gateway, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    terminal = release(root, receipt, bound, device_command, deadline['search_stop_at_epoch'])
                except NativeReleaseUnknown as error:
                    if kind == 'development':
                        from campaign.development_stop import halt
                        halt(bound, 'Native device release is unknown: ' + str(error), relative)
                    raise
                unchanged()
                if code or terminal['exit_code'] or terminal.get('status') != 'completed':
                    if kind == 'development':
                        from campaign.development_stop import device_failure_is_infrastructure, halt
                        reason = device_failure_is_infrastructure(folder)
                        if reason:
                            halt(bound, reason, relative)
                    raise ValueError(stage + ' failed; native evidence retained')
            if kind == 'development':
                outcome['stage'] = 'compare'
                unchanged()
                _, timeout = search_window(root, kind, bound, deadline)
                with (folder / 'compare.log').open('x') as log:
                    code = subprocess.call(['bash', 'scripts/dtk.sh', 'cpu', bound['image'],
                        '/usr/bin/timeout', '-k', '10s', str(timeout) + 's', 'python3',
                        'campaign/development_evaluate.py', '--phase', 'compare-native', '--id', identity,
                        '--native-manifest', relative + '/manifest.json'],
                        cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
                unchanged()
                if code or not read(folder / 'comparison.json')['numerically_passed']:
                    raise ValueError('Native original five-case comparison failed')
            if time.time() > deadline['search_stop_at_epoch']:
                raise ValueError('Native result completed after the original search cutoff')
            outcome['status'] = 'measurements_retained_pending_review'
    except (KeyError, TypeError, ValueError, OSError, subprocess.SubprocessError) as error:
        outcome.update(status='failed' if outcome.get('stage') else 'not_executed',
                       reason=type(error).__name__ + ': ' + str(error))
    outcome['completed_at_epoch'] = time.time()
    write_new(folder / 'outcome.json', outcome)
    return outcome


def author_context(root):
    rows = []
    for folder in sorted((root / 'campaign/engineering').glob('*')):
        if not folder.is_dir() or folder.is_symlink():
            continue
        try:
            value = read(folder / 'outcome.json')
            rows.append({key: value.get(key) for key in ('id', 'status', 'stage', 'reason')})
        except (OSError, ValueError):
            rows.append(dict(id=folder.name, status='unknown_incomplete'))
    return ('\nOwn native HIP explorations. These are engineering evidence, excluded from Cake nomination.\n'
            + json.dumps(dict(probes=rows[-20:], omitted=max(0, len(rows) - 20)), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    probe = sub.add_parser('run')
    probe.add_argument('--id', required=True)
    probe.add_argument('--request', required=True)
    probe.add_argument('--skip-reason', default='')
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--manifest', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare_native(ROOT, args.manifest)
        return 0
    result = run_probe(ROOT, args.id, args.request, args.skip_reason)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] in ('skipped', 'measurements_retained_pending_review') else 1


if __name__ == '__main__':
    sys.exit(main())
