#!/usr/bin/env python3
"""Prepare one host's new development Runs; never start an author or a GPU process."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from campaign.binding import GATEWAY_FILES, clean, git, read
from development.development_binding import KIND, validate_plan
from scripts.prepare_bench import clone, gateway_sources, new_json, source_identity


def prepare(plan_path, contracts_path, disposition_path, host, output):
    plan, contracts = read(plan_path), read(contracts_path)
    validate_plan(plan, contracts)
    disposition = read(disposition_path)
    if disposition.get('selected_compiler') != plan['compiler']:
        raise ValueError('The explicit Bench disposition must select this immutable Compiler')
    clean(ROOT, plan['adapter_commit'])
    identity = source_identity(ROOT)
    config = plan['hosts'][host]
    compiler, gateway = Path(config['compiler_root']), Path(config['gateway']['root'])
    clean(compiler, plan['compiler'])
    clean(gateway, config['gateway']['commit'])
    gateway_files = gateway_sources(gateway, config['gateway']['commit'])
    selected = [row for row in plan['assignments'] if row['host'] == host]
    if output.exists() or not selected or any(Path(row['root']).exists() for row in selected):
        raise ValueError('Preparation requires a nonempty host subset and fresh output/Run roots')
    if any(Path(row['root']).resolve() != Path(row['root']) for row in selected):
        raise ValueError('Run roots must be canonical paths without symlink aliases')
    seeds = {}
    for row in selected:
        source = Path(row['inherited']['source'])
        if not source.is_file() or source.is_symlink():
            raise ValueError('Inherited source must be a retained ordinary file')
        seeds[row['task']] = source.read_bytes()
        if not seeds[row['task']].strip():
            raise ValueError('Inherited source is empty')
    # This child isolates imports from the selected Compiler. It opens no device.
    subprocess.run([sys.executable, str(ROOT / 'development/catalog.py'), '--compiler-source',
                    str(compiler), '--compiler', plan['compiler'], '--contracts', str(contracts_path),
                    '--plan', str(plan_path), '--host', host], check=True)
    if any(Path(row['inherited']['source']).read_bytes() != seeds[row['task']] for row in selected):
        raise ValueError('Inherited material changed during CPU qualification')
    records = []
    for row in selected:
        task, run = row['task'], Path(row['root'])
        contract = contracts['tasks'][task]
        run.parent.mkdir(parents=True, exist_ok=True)
        clone(ROOT, run, plan['adapter_commit'])
        for key, value in identity.items():
            subprocess.run(['git', '-C', str(run), 'config', '--local', key, value], check=True)
        subprocess.run(['git', '-C', str(run), 'fetch', '--quiet', str(gateway), config['gateway']['commit']], check=True)
        for name, content in gateway_files.items():
            (run / name).write_bytes(content)
        (run / '.deps').mkdir()
        clone(compiler, run / '.deps/cake-ir', plan['compiler'])
        for name in ('candidates', 'evaluations', 'logs', 'JOURNAL', 'inherited'):
            (run / 'campaign' / name).mkdir()
        (run / '.local/triton-cache').mkdir(parents=True)
        for source in (ROOT / 'development').glob('*.py'):
            if source.name != 'catalog.py':
                shutil.copyfile(source, run / 'campaign' / source.name)
        binding = dict(task=task, root=str(run), host=host, home=config['home'], hcu=row['hcu'],
            compiler=plan['compiler'], gateway=config['gateway']['commit'], image=config['image'],
            gateway_files=list(GATEWAY_FILES), **plan['controls'],
            **{key: contract[key] for key in ('rows', 'columns', 'depth', 'case_ids', 'abi')},
            batch_stop_file=str(Path(config['owner_root']) / ('STOP-hcu' + str(row['hcu']) + '.json')))
        new_json(run / 'campaign/development-binding.json', binding)
        new_json(run / 'campaign/development-plan.json', plan)
        new_json(run / 'campaign/original-contracts.json', contracts)
        new_json(run / 'campaign/original-workload.json', contract['workload'])
        new_json(run / 'campaign/compiler-disposition.json', disposition)
        new_json(run / 'campaign/inherited/provenance.json', dict(row['inherited'], source='campaign/inherited/candidate.py'))
        (run / 'campaign/inherited/candidate.py').write_bytes(seeds[task])
        (run / 'campaign/budget-3h.yaml').write_text('budget:\n  hours: 3\n')
        policy = (ROOT / 'development/TASK.md').read_text()
        for key in ('task', 'rows', 'columns', 'depth', 'compiler'):
            policy = policy.replace('{' + key + '}', str(binding[key]))
        warning = ('The canonical gemm_bias starter has a retained mixed_magnitude failure; '
                   'preserve the five-case contract and report the result.' if task == 'gemm_bias' else '')
        (run / 'campaign/TASK.md').write_text(policy.replace('{starter_warning}', warning))
        (run / 'AGENTS.md').write_text('''# Registered Compiler-development Run
The pinned Compiler Task, original Workload, all five cases and comparator own semantics.
Read campaign/TASK.md, development-binding.json, deadline.json and profiling intake.
Use campaign/development_evaluate_owner.py and the qualified HCU gateway only.
The canonical starter and declared inherited candidate are separate inputs; old outcomes are not inherited.
The host owner assigns static author slots. Never modify its allocation, controls or terminal records.
Do not inspect Bench artifacts or another Run. Never touch HCU0 or an unassigned device.
''')
        (run / 'campaign/JOURNAL.md').write_text('# Own development evidence\n')
        (run / 'campaign/JOURNAL/EVOLUTION.md').write_text('# Owning-layer leads and No promotion\n')
        with (run / '.gitignore').open('a') as stream:
            for name in ('deadline.json', 'profile-intake.json', 'workload.json', 'development-inputs.pt',
                         'starter-assessment.json', 'candidates/', 'evaluations/', 'logs/', 'JOURNAL/',
                         'JOURNAL.md', 'ENDPOINT.json', 'DONE.json', 'controller-status.json',
                         'SEARCH_CLOSE.json', 'search-close-owner.json', 'search-close-owner.json.tmp',
                         'search-close-owner.jsonl', 'author-round-terminal.json', 'author-round-terminal.json.tmp'):
                stream.write('\ncampaign/' + name)
            stream.write('\n')
        subprocess.run(['git', '-C', str(run), 'add', '--', '.gitignore', 'AGENTS.md',
                        *GATEWAY_FILES, 'campaign'], check=True)
        subprocess.run(['git', '-C', str(run), 'commit', '-qm', 'Prepare immutable registered development Task'], check=True)
        records.append(dict(binding, prepared_commit=git(run, 'rev-parse', 'HEAD')))
    new_json(output, dict(kind=KIND, host=host, capacity_per_hcu=1,
        start_window_seconds=config['start_window_seconds'], runs=records))
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('plan', 'contracts', 'disposition', 'output'):
        parser.add_argument('--' + option, type=Path, required=True)
    parser.add_argument('--host', required=True)
    args = parser.parse_args()
    prepare(args.plan, args.contracts, args.disposition, args.host, args.output)
