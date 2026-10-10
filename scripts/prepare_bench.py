#!/usr/bin/env python3
"""Prepare fresh Run directories for one host; never launch authors or devices."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from campaign.binding import GATEWAY_FILES, clean, git, read, validate_plan, validate_routing

def gateway_sources(root, commit):
    sources = {}
    for name in GATEWAY_FILES:
        try:
            sources[name] = subprocess.check_output(['git', '-C', str(root), 'show', commit + ':' + name], stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as error:
            raise ValueError('Qualified gateway is missing required helper: ' + name) from error
    return sources


def source_identity(root):
    identity = {}
    for key in ('user.name', 'user.email'):
        try:
            value = git(root, 'config', '--get', key)
        except subprocess.CalledProcessError as error:
            raise ValueError('Configure the Bench source checkout Git ' + key + ' before preparation') from error
        if not value:
            raise ValueError('Bench source checkout Git ' + key + ' is empty')
        identity[key] = value
    return identity


def clone(source, destination, commit):
    subprocess.run(['git', 'clone', '--quiet', '--no-hardlinks', str(source), str(destination)], check=True)
    subprocess.run(['git', '-C', str(destination), 'checkout', '--quiet', '--detach', commit], check=True)


def new_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def prepare(plan_path, routing_path, host, output):
    plan, routing = read(plan_path), read(routing_path)
    validate_plan(plan, read(ROOT / 'suite.json'))
    bench_pin = plan['bindings']['bench']['commit']
    clean(ROOT, bench_pin)
    identity = source_identity(ROOT)
    if output.exists():
        raise FileExistsError(output)
    hosts = routing['hosts']
    if host not in hosts:
        raise ValueError('Host is absent from frozen routing')
    validate_routing(plan, routing)
    all_assignments = routing['assignments']
    config = hosts[host]
    gateway = Path(config['gateway']['root'])
    clean(gateway, config['gateway']['commit'])
    gateway_files = gateway_sources(gateway, config['gateway']['commit'])
    if not config['image'].startswith('sha256:'):
        raise ValueError('Use the qualified immutable image ID')
    if type(config['start_window_seconds']) is not int or config['start_window_seconds'] <= 0:
        raise ValueError('A positive host start window is required')
    if not 1 <= config['capacity_per_hcu'] <= 3:
        raise ValueError('Author capacity must be explicit and at most three per HCU')
    for condition in ('control', 'successor'):
        clean(Path(config['compiler_roots'][condition]), plan['bindings'][condition]['commit'])
    sol = Path(config['sol_root'])
    clean(sol, read(ROOT / 'sources.lock.json')['sol_execbench']['revision'])
    data = Path(config['data_root'])
    for task in plan['task_ids']:
        folder = data / 'benchmark' / task
        workloads = [json.loads(line) for line in (folder / 'workload.jsonl').read_text().splitlines() if line.strip()]
        if len(workloads) != 16 or len({row['uuid'] for row in workloads}) != 16:
            raise ValueError('Original task must contain sixteen unique workloads: ' + task)
        if not (folder / 'reference.py').is_file() or not (folder / 'definition.json').is_file():
            raise ValueError('Original task materialization is incomplete: ' + task)
    selected = [row for row in all_assignments if row['host'] == host]
    if any(Path(row['root']) != Path(row['root']).resolve() for row in selected):
        raise ValueError('Physical workspaces must use canonical paths without symlink aliases')
    if not selected or any(Path(row['root']).exists() for row in selected):
        raise ValueError('Expected nonempty fresh physical workspaces; no replacement or reset')
    allocation = {row['workspace']: row for row in plan['allocations']}
    records = []
    for route in selected:
        assignment = allocation[route['workspace']]
        run = Path(route['root'])
        run.parent.mkdir(parents=True, exist_ok=True)
        clone(ROOT, run, bench_pin)
        for key, value in identity.items():
            subprocess.run(['git', '-C', str(run), 'config', '--local', key, value], check=True)
        subprocess.run(['git', '-C', str(run), 'fetch', '--quiet', str(gateway), config['gateway']['commit']], check=True)
        for name, content in gateway_files.items():
            (run / name).write_bytes(content)
        (run / '.deps').mkdir()
        clone(Path(config['compiler_roots'][assignment['condition']]), run / '.deps/cake-ir', plan['bindings'][assignment['condition']]['commit'])
        clone(sol, run / '.deps/sol-execbench', read(ROOT / 'sources.lock.json')['sol_execbench']['revision'])
        shutil.copytree(data, run / '.data', ignore=shutil.ignore_patterns('cache', '__pycache__'))
        shutil.copytree(config['flag_gems_root'], run / '.deps/flag_gems_src_540_node2', ignore=shutil.ignore_patterns('__pycache__'))
        for name in ('candidates', 'evaluations', 'results', 'logs', 'schedules', 'management'):
            (run / 'campaign' / name).mkdir()
        for name in ('triton-cache', 'aiter-jit-cache'):
            (run / '.local' / name).mkdir(parents=True)
        binding = {**assignment, 'frozen_workspace': assignment['workspace'], 'root': str(run.resolve()),
                   'host': host, 'home': config['home'], 'hcu': route['hcu'], 'image': config['image'],
                   'author': plan['author'], 'bench_commit': bench_pin,
                   'compiler_commit': plan['bindings'][assignment['condition']]['commit'],
                   'gateway_commit': config['gateway']['commit'], 'gateway_files': list(GATEWAY_FILES)}
        binding.pop('workspace')
        new_json(run / 'campaign/frozen-plan.json', plan)
        new_json(run / 'campaign/frozen-routing.json', routing)
        new_json(run / 'campaign/binding.json', binding)
        baseline = plan['baseline_files'][assignment['task']]
        task = next(row for row in read(run / 'suite.json')['tasks'] if row['id'] == assignment['task'])
        group = {'arm': 'cake_ir', 'hcu': route['hcu'], 'compiler_commit': binding['compiler_commit'],
                 'tasks': [{'id': assignment['task'], 'baseline': baseline,
                            'baseline_sha256': hashlib.sha256((run / baseline).read_bytes()).hexdigest(),
                            'smoke_uuid': task['smoke_workload_uuid']}]}
        new_json(run / 'campaign/groups/c.json', group)
        (run / 'campaign/budget.yaml').write_text('budget:\n  hours: 3\n')
        (run / 'campaign/JOURNAL.md').write_text('# Own Bench search journal\n')
        (run / 'campaign/EVOLUTION.md').write_text('# Compiler leads from this Run\n')
        with (run / '.gitignore').open('a') as stream:
            stream.write('\n# Per-Run artifacts are retained locally, never source controls.\n')
            for name in ('candidates/', 'evaluations/', 'results/', 'logs/', 'schedules/', 'management/', 'generated/', 'final-confirmation/', 'deadline.json', 'intake.json', 'profile-intake.json', 'controller-status.json', 'ENDPOINT.json', 'DONE.json', 'JOURNAL.md', 'EVOLUTION.md'):
                stream.write('campaign/' + name + '\n')
        subprocess.run(['git', '-C', str(run), 'add', '.gitignore', *GATEWAY_FILES,
                        'campaign/frozen-plan.json', 'campaign/frozen-routing.json', 'campaign/binding.json', 'campaign/groups/c.json', 'campaign/budget.yaml'], check=True)
        subprocess.run(['git', '-C', str(run), 'commit', '-qm', 'Prepare frozen fresh Bench author allocation'], check=True)
        records.append({**binding, 'prepared_commit': git(run, 'rev-parse', 'HEAD')})
    # The queue consumes this create-only host subset in the global randomized order.
    records.sort(key=lambda row: next(i for i, item in enumerate(plan['allocations']) if item['workspace'] == row['frozen_workspace']))
    new_json(output, {'host': host, 'capacity_per_hcu': config['capacity_per_hcu'],
                      'start_window_seconds': config['start_window_seconds'], 'runs': records})
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--routing', required=True, type=Path)
    parser.add_argument('--host', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.plan, args.routing, args.host, args.output), indent=2))
