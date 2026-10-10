"""Reconcile prepared files with the immutable evolve allocation before intake."""
from pathlib import Path
import hashlib
import json
import subprocess


def read(path):
    return json.loads(path.read_text())


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def clean(root, commit):
    if git(root, 'rev-parse', 'HEAD') != commit or git(root, 'status', '--porcelain'):
        raise ValueError('Source is not the clean pinned commit: ' + str(root))


def validate_plan(plan, suite):
    if plan.get('kind') != 'rolling_bench_fresh_search' or plan.get('target') != 'gfx938':
        raise ValueError('Expected an evolve rolling gfx938 Bench allocation')
    tasks = [row['id'] for row in suite['tasks']]
    if len(set(plan['task_ids'])) != len(tasks) or set(plan['task_ids']) != set(tasks):
        raise ValueError('This launch must cover the complete expanded Bench suite')
    author = plan['author']
    if (author['wall_time_seconds'], author['confirmation_seconds']) != (10800, 1800):
        raise ValueError('Existing assay supports exactly 3 hours including 30 minute confirmation')
    if author['knowledge'] != 'none' or author['reference_access'] != 'known_kernel_reproduction':
        raise ValueError('This fresh cohort supports no additional knowledge and known-kernel reference access')
    if author['scaffold'] != 'bw1100-bench@' + plan['bindings']['bench']['commit'] + ':campaign/TASK.md':
        raise ValueError('Scaffold must name the frozen Bench TASK.md')
    if plan['measurement_protocol'] != 'campaign/protocol.json':
        raise ValueError('Use the maintained complete-call campaign protocol')
    allocations = plan['allocations']
    expected = {(task, condition, replicate) for task in tasks for condition in ('control', 'successor')
                for replicate in range(1, plan['replicates'] + 1)}
    actual = [(row['task'], row['condition'], row['replicate']) for row in allocations]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Allocation has missing, duplicate or unknown task/condition assignments')
    if len({row['workspace'] for row in allocations}) != len(allocations):
        raise ValueError('Allocation workspaces must be unique')
    if set(plan['baseline_files']) != set(tasks):
        raise ValueError('Every original task needs its fixed community baseline')
    if plan['total_author_wall_seconds'] < len(allocations) * author['wall_time_seconds']:
        raise ValueError('Frozen author budget does not fund every allocation')


def validate_routing(plan, routing):
    assignments, hosts = routing['assignments'], routing['hosts']
    if (len(assignments) != len(plan['allocations']) or
        {row['workspace'] for row in assignments} != {row['workspace'] for row in plan['allocations']} or
        len({row['workspace'] for row in assignments}) != len(assignments) or
        len({(row['host'], row['root']) for row in assignments}) != len(assignments)):
        raise ValueError('Routing must cover each frozen workspace exactly once')
    by_workspace = {row['workspace']: row for row in assignments}
    pairs = {}
    for allocation in plan['allocations']:
        row = by_workspace[allocation['workspace']]
        if (row['host'] not in hosts or type(row['hcu']) is not int or row['hcu'] not in range(1, 7)
                or not Path(row['root']).is_absolute()):
            raise ValueError('Unknown host, unsupported HCU or relative physical workspace')
        pair = (allocation['task'], allocation['replicate'])
        location = (row['host'], row['hcu'])
        if pair in pairs and pairs[pair] != location:
            raise ValueError('Both Compiler conditions of a task must share host and HCU')
        pairs[pair] = location
    return by_workspace


def reconcile(root):
    binding = read(root / 'campaign/binding.json')
    plan = read(root / 'campaign/frozen-plan.json')
    suite = read(root / 'suite.json')
    validate_plan(plan, suite)
    routing = read(root / 'campaign/frozen-routing.json')
    routes = validate_routing(plan, routing)
    route = routes[binding['frozen_workspace']]
    config = routing['hosts'][route['host']]
    if any(binding[field] != route[field] for field in ('host', 'root', 'hcu')):
        raise ValueError('Prepared route differs from the frozen global routing')
    if binding['image'] != config['image'] or binding['home'] != config['home'] or binding['gateway_commit'] != config['gateway']['commit']:
        raise ValueError('Prepared runtime binding differs from frozen host configuration')
    selected = [row for row in plan['allocations'] if row['workspace'] == binding['frozen_workspace']]
    if len(selected) != 1:
        raise ValueError('Run must bind one exact frozen workspace')
    selected = selected[0]
    for field in ('task', 'condition', 'replicate'):
        if selected[field] != binding[field]:
            raise ValueError('Prepared assignment differs: ' + field)
    if binding['author'] != plan['author']:
        raise ValueError('Prepared author controls differ')
    if binding['compiler_commit'] != plan['bindings'][selected['condition']]['commit']:
        raise ValueError('Prepared Compiler differs')
    if str(root.resolve()) != binding['root']:
        raise ValueError('Prepared physical root differs from host routing')
    if binding['hcu'] not in range(1, 7):
        raise ValueError('Only declared HCU1..6 routes are supported')
    clean(root / '.deps/cake-ir', binding['compiler_commit'])
    clean(root, git(root, 'rev-parse', 'HEAD'))
    bench = plan['bindings']['bench']['commit']
    for relative in git(root, 'ls-tree', '-r', '--name-only', bench).splitlines():
        if relative in binding['gateway_files'] or relative == '.gitignore':
            continue
        expected = subprocess.check_output(['git', '-C', str(root), 'show', bench + ':' + relative])
        if (root / relative).read_bytes() != expected:
            raise ValueError('Frozen Bench source changed: ' + relative)
    for relative in binding['gateway_files']:
        expected = subprocess.check_output(['git', '-C', str(root), 'show', binding['gateway_commit'] + ':' + relative])
        if (root / relative).read_bytes() != expected:
            raise ValueError('Qualified host gateway changed: ' + relative)
    group = read(root / 'campaign/groups/c.json')
    baseline = plan['baseline_files'][binding['task']]
    expected_task = next(row for row in suite['tasks'] if row['id'] == binding['task'])
    if group != {'arm': 'cake_ir', 'hcu': binding['hcu'], 'compiler_commit': binding['compiler_commit'],
                 'tasks': [{'id': binding['task'], 'baseline': baseline,
                            'baseline_sha256': hashlib.sha256((root / baseline).read_bytes()).hexdigest(),
                            'smoke_uuid': expected_task['smoke_workload_uuid']}]}:
        raise ValueError('Prepared evaluation group differs from frozen task/reference/route')
    return binding
