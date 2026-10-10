"""CPU fixtures for development preparation and the shared finite author owner."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from development.development_binding import reconcile, validate_plan
from scripts import bench_queue as queue, prepare_development as prepare


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True,
                                   stderr=subprocess.DEVNULL).strip()


def initialize(root):
    git(root, 'init', '-q')
    git(root, 'config', 'user.name', 'CPU Fixture')
    git(root, 'config', 'user.email', 'cpu-fixture@example.invalid')
    git(root, 'add', '.')
    git(root, 'commit', '-qm', 'CPU fixture source')
    return git(root, 'rev-parse', 'HEAD')


CASES = ['primary', 'zeros', 'near_zero', 'alternating', 'mixed_magnitude']
ABI = [dict(name='x', dtype='fp32', shape=[1], mode='input')]


def document(task):
    return dict(task=task, rows=1, columns=1, depth=1, case_ids=CASES, abi=ABI)


def allocation(root, adapter='a' * 40, compiler='b' * 40):
    tasks = {'task_' + str(i): dict(rows=1, columns=1, depth=1, case_ids=CASES,
              abi=ABI, workload=document('task_' + str(i))) for i in range(53)}
    controls = dict(model='claude/glm-5.3:high', wall_time_seconds=10800,
        search_seconds=9000, confirmation_seconds=1800, token_limits=None, capacity_per_hcu=1,
        reference_access='known_kernel_reproduction_with_declared_development_incumbent')
    hosts = {name: dict(hcus=[1, 2] if name == 'first' else [2, 3], home=str(root / 'home'),
        owner_root=str(root / name / 'owner'), compiler_root=str(root / 'compiler'),
        gateway=dict(root=str(root / 'adapter'), commit=adapter), image='sha256:' + 'c' * 64,
        start_window_seconds=172800) for name in ('first', 'second')}
    rows = [dict(task=task, host='first' if i < 2 else 'second', hcu=(1 + i if i < 2 else 2),
        root=str(root / 'runs' / task), inherited=dict(source=str(root / 'seed.py'),
            compiler='d' * 40, candidate_id='old-nominee', candidate_path='/historical/candidate.py',
            endpoint_path='/historical/ENDPOINT.json')) for i, task in enumerate(tasks)]
    plan = dict(kind='registered_task_development', target='gfx938', compiler=compiler,
                adapter_commit=adapter, controls=controls, hosts=hosts, assignments=rows)
    return plan, dict(source_compiler='e' * 40, tasks=tasks)


class DevelopmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_static_contract_and_explicit_version(self):
        plan, contracts = allocation(self.root)
        validate_plan(plan, contracts)
        for change in ('duplicate', 'missing', 'unselected', 'hcu_zero', 'host_excluded', 'two_authors', 'budget'):
            bad = copy.deepcopy(plan)
            if change == 'duplicate': bad['assignments'][-1] = bad['assignments'][0]
            if change == 'missing': bad['assignments'].pop()
            if change == 'unselected': bad['compiler'] = None
            if change == 'hcu_zero': bad['assignments'][0]['hcu'] = 0
            if change == 'host_excluded': bad['assignments'][2]['hcu'] = 1
            if change == 'two_authors': bad['controls']['capacity_per_hcu'] = 2
            if change == 'budget': bad['controls']['search_seconds'] = 9100
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_plan(bad, contracts)

    def sources(self):
        adapter = self.root / 'adapter'
        adapter.mkdir()
        for name in ('campaign', 'development', 'scripts'):
            shutil.copytree(ROOT / name, adapter / name, ignore=shutil.ignore_patterns('__pycache__'))
        (adapter / 'scripts/hcu_device_identity.py').write_text('# CPU gateway fixture\n')
        shutil.copyfile(ROOT / '.gitignore', adapter / '.gitignore')
        compiler = self.root / 'compiler'
        package = compiler / 'src/open_cake_ir'
        (package / 'compiler').mkdir(parents=True)
        (package / 'tasks').mkdir()
        (package / '__init__.py').write_text('')
        (package / 'tasks/__init__.py').write_text('')
        (compiler / 'compiler').mkdir()
        (compiler / 'compiler/revision.json').write_text('{}')
        (package / 'compiler/__init__.py').write_text('''from types import SimpleNamespace
import subprocess
class Compiler:
 @classmethod
 def load(cls, root, revision):
  value=cls();value.commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip();return value
 def assess(self, document):
  buffer=SimpleNamespace(name='x',shape=(1,),dtype=SimpleNamespace(value='fp32'),mode=SimpleNamespace(value='input'),space=SimpleNamespace(value='global'))
  return SimpleNamespace(lowering_eligible=True,target='gfx938',typed_schedule=SimpleNamespace(buffers=[buffer]))
 def lower(self, assessment):return SimpleNamespace(source='generated')
frontend=SimpleNamespace(parse=lambda source:SimpleNamespace(document={}))
''')
        (package / 'tasks/workloads.py').write_text('''from types import SimpleNamespace
CASES=['primary','zeros','near_zero','alternating','mixed_magnitude']
ABI=[dict(name='x',dtype='fp32',shape=[1],mode='input')]
def create_task(task, backend, rows, columns, depth):
 return dict(task=task,rows=rows,columns=columns,depth=depth,case_ids=CASES,abi=ABI),'starter'
class WorkloadContract:
 def __init__(self, document):self.document=document;self.case_ids=document['case_ids']
 def tensor_abi(self, case):return [SimpleNamespace(**row) for row in self.document['abi']]
''')
        adapter_pin, compiler_pin = initialize(adapter), initialize(compiler)
        plan, contracts = allocation(self.root, adapter_pin, compiler_pin)
        (self.root / 'seed.py').write_text('# retained historical source, not a qualification\n')
        write(self.root / 'plan.json', plan)
        write(self.root / 'contracts.json', contracts)
        write(self.root / 'disposition.json', {'selected_compiler': compiler_pin, 'basis': 'CPU fixture'})
        return adapter, plan, contracts

    def prepare(self, adapter):
        with patch.object(prepare, 'ROOT', adapter):
            return prepare.prepare(self.root / 'plan.json', self.root / 'contracts.json',
                self.root / 'disposition.json', 'first', self.root / 'allocation.json')

    def test_real_git_preparation_preserves_separate_material_and_contract(self):
        adapter, plan, contracts = self.sources()
        rows = self.prepare(adapter)
        self.assertEqual(len(rows), 2)
        for row in rows:
            run = Path(row['root'])
            self.assertEqual(reconcile(run)['compiler'], plan['compiler'])
            self.assertEqual(json.loads((run / 'campaign/original-workload.json').read_text()),
                             contracts['tasks'][row['task']]['workload'])
            self.assertEqual((run / 'campaign/inherited/candidate.py').read_bytes(), (self.root / 'seed.py').read_bytes())
            self.assertFalse((run / 'campaign/candidates/starter.py').exists())
            self.assertFalse(list((run / 'campaign/evaluations').iterdir()))
            self.assertFalse((run / 'campaign/deadline.json').exists())
            self.assertFalse((run / 'campaign/DONE.json').exists())
        with self.assertRaises(ValueError): self.prepare(adapter)
        # Real --check uses an unusable sibling hmz; failure must precede budget/intake.
        bin_path = self.root / 'venv/bin'
        subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(bin_path.parent)], check=True)
        (bin_path / 'hmz').write_text('#!/bin/sh\nexit 2\n')
        (bin_path / 'hmz').chmod(0o755)
        result = subprocess.run([str(bin_path / 'python'), 'campaign/development_launch.py', '--check'],
                                cwd=rows[0]['root'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn('CalledProcessError', result.stderr)
        self.assertFalse((Path(rows[0]['root']) / 'campaign/deadline.json').exists())
        self.assertFalse((Path(rows[0]['root']) / 'campaign/intake.json').exists())
        run = Path(rows[0]['root'])
        for name in ('campaign/development_control.py', 'campaign/launch.py', 'campaign/TASK.md', 'AGENTS.md'):
            original = (run / name).read_text()
            (run / name).write_text(original + '\n# changed after preparation\n')
            git(run, 'add', name)
            git(run, 'commit', '-qm', 'Changed frozen source fixture')
            self.assertFalse(git(run, 'diff', '--name-only', 'HEAD'))
            with self.assertRaisesRegex(ValueError, 'adapter changed|author source changed|scaffold changed|instructions'):
                reconcile(run)
            (run / name).write_text(original)
            git(run, 'add', name)
            git(run, 'commit', '-qm', 'Restore source fixture')

    def test_whole_workload_drift_or_unselected_disposition_creates_no_run(self):
        adapter, plan, contracts = self.sources()
        write(self.root / 'disposition.json', {'selected_compiler': None})
        with self.assertRaises(ValueError): self.prepare(adapter)
        self.assertFalse((self.root / 'runs').exists())
        write(self.root / 'disposition.json', {'selected_compiler': plan['compiler']})
        contracts['tasks']['task_52']['workload']['oracle_change'] = True
        write(self.root / 'contracts.json', contracts)
        with self.assertRaises(subprocess.CalledProcessError): self.prepare(adapter)
        self.assertFalse((self.root / 'runs').exists())

    def test_real_queue_keeps_failed_receipt_and_allows_other_hcu(self):
        # Only the children are CPU fixture scripts; the Git checks and queue loop run.
        rows = []
        compiler = self.root / 'compiler'
        compiler.mkdir()
        (compiler / 'source.py').write_text('pass\n')
        compiler_pin = initialize(compiler)
        for index in range(3):
            run = self.root / ('r' + str(index))
            (run / 'campaign/logs').mkdir(parents=True)
            (run / '.deps').mkdir()
            subprocess.run(['git', 'clone', '-q', str(compiler), str(run / '.deps/cake-ir')], check=True)
            binding = dict(task='task_' + str(index), root=str(run), host='fixture',
                           hcu=1 if index < 2 else 2, compiler=compiler_pin,
                           batch_stop_file=str(self.root / 'owner' / ('STOP-hcu' + str(1 if index < 2 else 2) + '.json')))
            write(run / 'campaign/development-binding.json', binding)
            (run / '.gitignore').write_text('.deps/\ncampaign/logs/\ncampaign/admission*.json\ncampaign/DONE.json\n')
            (run / 'campaign/development_launch.py').write_text('''import json,sys
from pathlib import Path
if sys.argv[1:]==['--check']:raise SystemExit(0)
b=json.loads(Path('campaign/development-binding.json').read_text())
a=dict(schema='bw1100-bench.hcu-admission.v1',job_id=b['task'])
Path('campaign/admission.json').write_text(json.dumps(a))
if b['hcu']==2:
 a.update(completed_at=1,exit_code=0,container_still_running=False,after_vram='0%',after_kfd_visible=False)
 Path('campaign/admission-terminal.json').write_text(json.dumps(a))
Path('campaign/DONE.json').write_text(json.dumps(dict(terminal=True,status='completed',device_release_verified=b['hcu']==2)))
''')
            rows.append(dict(binding, prepared_commit=initialize(run)))
        path = self.root / 'allocation.json'
        write(path, dict(kind='registered_task_development', host='fixture',
                         capacity_per_hcu=1, start_window_seconds=1, runs=rows))
        # Short polling preserves real process and wall-clock behavior.
        original_sleep = queue.time.sleep
        with patch.object(queue.time, 'sleep', side_effect=lambda _: original_sleep(0.02)):
            queue.run(self.root / 'owner', path)
        status = json.loads((self.root / 'owner/STATUS.json').read_text())
        self.assertEqual(status['pending'], ['task_1'])
        self.assertEqual({row['task'] for row in status['finished']}, {'task_0', 'task_2'})
        self.assertIn('1', status['blocked_hcus'])
        self.assertNotIn('2', status['blocked_hcus'])
        with self.assertRaises(AssertionError): queue.run(self.root / 'owner', path)


if __name__ == '__main__':
    unittest.main()
