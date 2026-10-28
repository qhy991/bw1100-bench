#!/usr/bin/env python3
"""Prepare one host's new development Runs; never start an author or a GPU process."""
from pathlib import Path
import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from campaign.binding import GATEWAY_FILES, clean, git, read
from development.development_binding import ADAPTER_SCRIPTS, BUDGET, KIND, inherited_path, render_agents, render_task, validate_plan
from scripts.json_stream import RECEIPT_REFUSED_DIRS, RELEASE_ROOTS
from scripts.prepare_bench import clone, gateway_sources, new_json, source_identity

# What the pinned gateway's _receipt_paths must refuse and admit (#44, #45 review).
RECEIPT_REFUSED = ('campaign/__pycache__/p.json', '.data/p.json', 'native/p.json', 'campaign/x-terminal.json',
                   *('%s/%s/p.json' % (top, cache) for top in RELEASE_ROOTS for cache in sorted(RECEIPT_REFUSED_DIRS)))
RECEIPT_ADMITTED = ('campaign/evaluations/e/p.json', '.local/profile/p.json')


def require_receipt_scope(gateway_source, commit):
    """The Run executes the pinned gateway; it must refuse every receipt the checks may miss.

    The source is imported as an isolated module (its main() never runs) with ROOT bound to
    an empty temporary directory, and its own _receipt_paths is exercised, so a guard that is
    declared but not enforced is refused like a missing one.
    """
    refusal = 'Pinned gateway ' + commit + ' scripts/hcu_run.py lacks the receipt-scope guard: '
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        (root / 'scripts').mkdir()
        (root / 'scripts/hcu_run.py').write_bytes(gateway_source)
        spec = importlib.util.spec_from_file_location('pinned_gateway', str(root / 'scripts/hcu_run.py'))
        gateway = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(gateway)
        except Exception as error:
            raise ValueError(refusal + 'it does not import: %r' % error) from error
        if (getattr(gateway, 'RECEIPT_ROOTS', None), getattr(gateway, 'RECEIPT_UNWALKED_DIRS', None)) != (
                RELEASE_ROOTS, RECEIPT_REFUSED_DIRS):
            raise ValueError(refusal + 'RECEIPT_ROOTS/RECEIPT_UNWALKED_DIRS differ from the release checks\' copy')
        gateway.ROOT = root
        for name in RECEIPT_REFUSED:
            try:
                gateway._receipt_paths(name)
            except ValueError:
                continue
            except Exception as error:
                raise ValueError(refusal + '%s raised %r instead of a refusal' % (name, error)) from error
            raise ValueError(refusal + 'it admits ' + name)
        for name in RECEIPT_ADMITTED:
            try:
                gateway._receipt_paths(name)
            except Exception as error:
                raise ValueError(refusal + 'it refuses ' + name + ': %r' % error) from error


def prepare(plan_path, contracts_path, disposition_path, host, output):
    plan, contracts = read(plan_path), read(contracts_path)
    validate_plan(plan, contracts)
    disposition = read(disposition_path)
    if disposition.get('selected_compiler') != plan['compiler']:
        raise ValueError('The explicit engineering selection must select this immutable Compiler')
    clean(ROOT, plan['adapter_commit'])
    identity = source_identity(ROOT)
    config = plan['hosts'][host]
    compiler, gateway = Path(config['compiler_root']), Path(config['gateway']['root'])
    clean(compiler, plan['compiler'])
    clean(gateway, config['gateway']['commit'])
    gateway_files = gateway_sources(gateway, config['gateway']['commit'])
    require_receipt_scope(gateway_files['scripts/hcu_run.py'], config['gateway']['commit'])
    selected = [row for row in plan['assignments'] if row['host'] == host]
    if output.exists() or not selected or any(Path(row['root']).exists() for row in selected):
        raise ValueError('Preparation requires a nonempty host subset and fresh output/Run roots')
    if any(Path(row['root']).resolve() != Path(row['root']) for row in selected):
        raise ValueError('Run roots must be canonical paths without symlink aliases')
    seeds = {}
    for row in selected:
        if row.get('inherited') is None:
            continue
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
    if any(Path(row['inherited']['source']).read_bytes() != seeds[row['task']]
           for row in selected if row.get('inherited') is not None):
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
        for name in ADAPTER_SCRIPTS:
            shutil.copyfile(ROOT / 'development' / name, run / 'campaign' / name)
        binding = dict(task=task, root=str(run), host=host, home=config['home'], hcu=row['hcu'],
            hcu_pci=config['hcu_pci'][str(row['hcu'])],
            compiler=plan['compiler'], gateway=config['gateway']['commit'], image=config['image'],
            gateway_files=list(GATEWAY_FILES), **plan['controls'],
            **{key: contract[key] for key in ('rows', 'columns', 'depth', 'case_ids', 'abi')},
            batch_stop_file=str(Path(config['owner_root']) / ('STOP-hcu' + str(row['hcu']) + '.json')))
        if 'hypotheses' in plan:
            binding['hypothesis'] = plan['hypotheses'][task]
        new_json(run / 'campaign/development-binding.json', binding)
        new_json(run / 'campaign/development-plan.json', plan)
        new_json(run / 'campaign/original-contracts.json', contracts)
        new_json(run / 'campaign/original-workload.json', contract['workload'])
        new_json(run / 'campaign/compiler-disposition.json', disposition)
        if row.get('inherited') is not None:
            new_json(run / 'campaign/inherited/provenance.json', dict(row['inherited'], source=inherited_path(row)))
            (run / inherited_path(row)).write_bytes(seeds[task])
        (run / 'campaign/budget-3h.yaml').write_text(BUDGET)
        (run / 'campaign/TASK.md').write_text(render_task((ROOT / 'development/TASK.md').read_text(), binding))
        (run / 'AGENTS.md').write_text(render_agents())
        (run / 'campaign/JOURNAL.md').write_text('# Own development evidence\n')
        (run / 'campaign/JOURNAL/EVOLUTION.md').write_text('# Owning-layer leads and No promotion\n')
        subprocess.run([sys.executable, str(run / 'campaign/compiler_tools.py'), 'freeze'], check=True)
        with (run / '.gitignore').open('a') as stream:
            for name in ('deadline.json', 'profile-intake.json', 'workload.json', 'development-inputs.pt',
                         'starter-assessment.json', 'compiler-actions/', 'candidates/', 'evaluations/', 'engineering/', 'logs/', 'JOURNAL/',
                         'JOURNAL.md', 'ENDPOINT.json', 'DONE.json', 'controller-status.json',
                         'SEARCH_CLOSE.json', 'search-close-owner.json', 'search-close-owner.json.tmp',
                         'search-close-owner.jsonl', 'author-round-terminal.json', 'author-round-terminal.json.tmp'):
                stream.write('\ncampaign/' + name)
            stream.write('\n')
        subprocess.run(['git', '-C', str(run), 'add', '--', '.gitignore', 'AGENTS.md',
                        *GATEWAY_FILES, 'campaign'], check=True)
        subprocess.run(['git', '-C', str(run), 'commit', '-qm', 'Prepare immutable registered development Task'], check=True)
        records.append(dict(binding, prepared_commit=git(run, 'rev-parse', 'HEAD')))
    new_json(output, dict(kind=KIND, host=host, capacity_per_hcu=plan['controls']['capacity_per_hcu'],
        start_window_seconds=config['start_window_seconds'], runs=records))
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('plan', 'contracts', 'disposition', 'output'):
        parser.add_argument('--' + option, type=Path, required=True)
    parser.add_argument('--host', required=True)
    args = parser.parse_args()
    prepare(args.plan, args.contracts, args.disposition, args.host, args.output)
