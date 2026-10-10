"""The prepared development binding, distinct from the independent Bench contract."""
from pathlib import Path
import json
import re
import subprocess

KIND = 'registered_task_development'
ADAPTER_SCRIPTS = ('development_binding.py', 'development_control.py', 'development_device.py',
                   'development_evaluate.py', 'development_evaluate_owner.py', 'development_launch.py',
                   'development_prepare.py', 'development_stop.py', 'ralph_flow.py')
SHARED_SCRIPTS = ('campaign/launch.py', 'campaign/binding.py', 'scripts/bench_queue.py',
                  'scripts/ralph_profile_intake.py')
BUDGET = 'budget:\n  hours: 3\n'


def render_task(template, binding):
    for key in ('task', 'rows', 'columns', 'depth', 'compiler'):
        template = template.replace('{' + key + '}', str(binding[key]))
    warning = ('The canonical gemm_bias starter has a retained mixed_magnitude failure; '
               'preserve the five-case contract and report the result.' if binding['task'] == 'gemm_bias' else '')
    return template.replace('{starter_warning}', warning)


def render_agents():
    return '''# Registered Compiler-development Run
The pinned Compiler Task, original Workload, all five cases and comparator own semantics.
Read campaign/TASK.md, development-binding.json, deadline.json and profiling intake.
Use campaign/development_evaluate_owner.py and the qualified HCU gateway only.
The canonical starter and declared inherited candidate are separate inputs; old outcomes are not inherited.
The host owner assigns static author slots. Never modify its allocation, controls or terminal records.
Do not inspect Bench artifacts or another Run. Never touch HCU0 or an unassigned device.
'''


def read(path):
    return json.loads(path.read_text())


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def source_bytes(root, commit, name):
    return subprocess.check_output(['git', '-C', str(root), 'show', commit + ':' + name])


def validate_plan(plan, contracts):
    if plan.get('kind') != KIND or plan.get('target') != 'gfx938':
        raise ValueError('Expected a registered gfx938 development allocation')
    for field in ('compiler', 'adapter_commit'):
        if not isinstance(plan.get(field), str) or not re.fullmatch('[0-9a-f]{40}', plan[field]):
            raise ValueError('Select an immutable ' + field + ' after the Bench disposition')
    controls = plan['controls']
    if (controls['wall_time_seconds'], controls['search_seconds'], controls['confirmation_seconds']) != (10800, 9000, 1800):
        raise ValueError('Preserve the three-hour budget including thirty-minute confirmation')
    if controls['token_limits'] is not None or controls['capacity_per_hcu'] != 1:
        raise ValueError('Development keeps one author per HCU and accounting-only tokens')
    if controls['reference_access'] != 'known_kernel_reproduction_with_declared_development_incumbent':
        raise ValueError('Declare the canonical starter and inherited development material')
    tasks = contracts['tasks']
    rows = plan['assignments']
    if len(tasks) != 53 or len(rows) != 53 or {row['task'] for row in rows} != set(tasks):
        raise ValueError('Assign every original development task exactly once')
    if len({row['root'] for row in rows}) != len(rows):
        raise ValueError('Every development Run needs a unique fresh root')
    for host, config in plan['hosts'].items():
        hcus = config['hcus']
        if (not hcus or len(set(hcus)) != len(hcus) or
                any(type(hcu) is not int or hcu not in range(1, 7) for hcu in hcus)):
            raise ValueError('Use the explicitly qualified nonzero HCU subset')
        if type(config['start_window_seconds']) is not int or config['start_window_seconds'] <= 0:
            raise ValueError('Declare a bounded host start window')
        for field in ('home', 'owner_root', 'compiler_root'):
            if not Path(config[field]).is_absolute():
                raise ValueError('Host paths must be absolute')
        if not re.fullmatch('sha256:[0-9a-f]{64}', config['image']):
            raise ValueError('Use the qualified immutable image')
    for row in rows:
        config = plan['hosts'][row['host']]
        if type(row['hcu']) is not int or row['hcu'] not in config['hcus'] or not Path(row['root']).is_absolute():
            raise ValueError('Assignment is outside its declared host HCU set')
        contract = tasks[row['task']]
        if len(contract['case_ids']) != 5 or len(set(contract['case_ids'])) != 5:
            raise ValueError('Preserve all five original cases')
        material = row['inherited']
        if not Path(material['source']).is_absolute() or not re.fullmatch('[0-9a-f]{40}', material['compiler']):
            raise ValueError('Declare an exact historical source material and Compiler')
        if not all(isinstance(material.get(key), str) and material[key]
                   for key in ('candidate_id', 'candidate_path', 'endpoint_path')):
            raise ValueError('Inherited source needs its historical candidate and endpoint provenance')


def reconcile(root, mounted=False):
    root = root.resolve()
    # Generated candidates and records are ignored; the prepared controls stay tracked.
    if git(root, 'diff', '--name-only', 'HEAD'):
        raise ValueError('Frozen development source or controls were modified')
    deadline = root / 'campaign/deadline.json'
    if deadline.exists() and read(deadline).get('source_commit') != git(root, 'rev-parse', 'HEAD'):
        raise ValueError('Prepared source changed after Run intake')
    binding = read(root / 'campaign/development-binding.json')
    plan = read(root / 'campaign/development-plan.json')
    original = read(root / 'campaign/original-contracts.json')
    validate_plan(plan, original)
    adapter = plan['adapter_commit']
    for name in ADAPTER_SCRIPTS:
        if (root / 'campaign' / name).read_bytes() != source_bytes(root, adapter, 'development/' + name):
            raise ValueError('Frozen development adapter changed: ' + name)
    for name in SHARED_SCRIPTS:
        if (root / name).read_bytes() != source_bytes(root, adapter, name):
            raise ValueError('Frozen shared author source changed: ' + name)
    row = next(item for item in plan['assignments'] if item['task'] == binding['task'])
    config = plan['hosts'][row['host']]
    if mounted and root != Path('/work'):
        raise ValueError('The qualified container mounts the Run at /work')
    if ((not mounted and root != Path(row['root'])) or binding['root'] != row['root'] or
            any(binding[field] != row[field] for field in ('task', 'host', 'hcu'))):
        raise ValueError('Prepared Run differs from its frozen static allocation')
    if binding['compiler'] != plan['compiler'] or binding['gateway'] != config['gateway']['commit']:
        raise ValueError('Prepared Compiler or qualified gateway changed')
    if read(root / 'campaign/compiler-disposition.json').get('selected_compiler') != binding['compiler']:
        raise ValueError('Compiler differs from the explicit Bench disposition')
    if binding['home'] != config['home'] or binding['image'] != config['image']:
        raise ValueError('Prepared host environment changed')
    if binding['batch_stop_file'] != str(Path(config['owner_root']) / ('STOP-hcu' + str(row['hcu']) + '.json')):
        raise ValueError('Stop file must belong to this HCU under its host owner')
    contract = original['tasks'][binding['task']]
    if any(binding[key] != contract[key] for key in ('rows', 'columns', 'depth', 'case_ids', 'abi')):
        raise ValueError('Prepared task differs from its original shape, cases or ABI')
    if read(root / 'campaign/original-workload.json') != contract['workload']:
        raise ValueError('Prepared Workload differs from the original contract')
    if any(binding[key] != plan['controls'][key] for key in plan['controls']):
        raise ValueError('Prepared author controls differ from the frozen plan')
    template = source_bytes(root, adapter, 'development/TASK.md').decode()
    if (root / 'campaign/TASK.md').read_text() != render_task(template, binding):
        raise ValueError('Frozen development author scaffold changed')
    if (root / 'AGENTS.md').read_text() != render_agents() or (root / 'campaign/budget-3h.yaml').read_text() != BUDGET:
        raise ValueError('Frozen development instructions or budget changed')
    compiler = root / '.deps/cake-ir'
    if git(compiler, 'rev-parse', 'HEAD') != binding['compiler'] or git(compiler, 'status', '--porcelain'):
        raise ValueError('The selected Compiler must remain clean and immutable')
    from campaign.binding import GATEWAY_FILES
    if binding['gateway_files'] != list(GATEWAY_FILES):
        raise ValueError('Qualified gateway helper closure changed')
    for name in GATEWAY_FILES:
        expected = source_bytes(root, binding['gateway'], name)
        if (root / name).read_bytes() != expected:
            raise ValueError('Qualified gateway file changed: ' + name)
    material = dict(row['inherited'], source='campaign/inherited/candidate.py')
    if read(root / 'campaign/inherited/provenance.json') != material:
        raise ValueError('Inherited source provenance changed')
    return binding
