"""CPU custody and command-boundary fixtures; no HIP build or GPU qualification."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from campaign import hip_probe as probe
from campaign import final_confirmation
from development import development_control, development_stop


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def source_fixture(root):
    source = root / 'campaign/candidates/native'
    source.mkdir(parents=True)
    (source / 'kernel.hip').write_text('// CPU fixture retains bytes; it does not compile HIP.\n')
    (source / 'entry.py').write_text('''import os
from pathlib import Path
def prepare():
 cache=Path(os.environ['HIP_PROBE_BUILD_DIR'])
 (cache/'fixture.bin').write_bytes(b'CPU fixture artifact')
 return {'artifacts':['fixture.bin'],'metadata':{'compiler':'CPU fixture'}}
def run(x):
 return x
''')
    write(root / 'campaign/candidates/gap.json', dict(diagnostic='concrete fixture limitation'))
    request = dict(target='gfx938', source_dir='campaign/candidates/native', entry='entry.py',
        build_flags=['--offload-arch=gfx938'], gap=dict(hypothesis='fixture mechanism',
        operation='fixture operation', limitation='retained fixture limitation',
        supported_transforms='fixture transform cannot express the proposed mechanism',
        proposed_owner='IR vocabulary pending review', counterexamples='all device behavior remains untested',
        evidence=['campaign/candidates/gap.json']))
    write(root / 'campaign/candidates/request.json', request)
    return request


class NativeProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.request = source_fixture(self.root)
        self.bound = dict(task='L1/fixture', image='sha256:' + 'a' * 64, hcu=4,
                          compiler_commit='b' * 40)
        started = time.time() - 5400
        self.deadline = dict(started_at_epoch=started, search_stop_at_epoch=started + 9000,
                             stop_at_epoch=started + 10800)
        write(self.root / 'campaign/deadline.json', self.deadline)
        self.calls = []
        self.damage = None
        self.failure = None

    def tearDown(self):
        self.temp.cleanup()

    def execute(self, argv, **kwargs):
        self.calls.append(argv)
        if argv[:3] == ['bash', 'scripts/dtk.sh', 'cpu']:
            if self.failure == 'build':
                return 7
            probe.prepare_native(self.root, argv[-1])
            if self.damage == 'source_during_prepare':
                (self.root / 'campaign/engineering/probe/source/kernel.hip').write_text('changed')
            return 0
        self.assertEqual(argv[:2], ['python3', 'scripts/hcu_run.py'])
        self.assertEqual(kwargs['cwd'], self.root)
        receipt_name = argv[argv.index('--receipt') + 1]
        stage = Path(receipt_name).name.split('-')[0]
        if self.failure == 'missing_admission':
            return 0
        if self.failure in ('lock_wait', 'lock_wait_with_terminal') or str(self.failure).startswith('refusal:'):
            reason = self.failure[len('refusal:'):] if str(self.failure).startswith('refusal:') else development_stop.LOCK_WAIT_TIMEOUT
            kwargs['stdout'].write('RuntimeError: ' + reason + '\n')
            if self.failure == 'lock_wait_with_terminal':
                write(self.root / receipt_name.replace('.json', '-terminal.json'), {'status': 'completed'})
            return 1
        command = argv[argv.index('--') + 1:]
        receipt = self.root / receipt_name
        before = dict(schema=probe.ADMISSION, job_id='fixture-' + stage,
            selected_hcu=self.bound['hcu'], image_id=self.bound['image'],
            command=command, started_at=time.time(), timeout_s=900)
        write(receipt, before)
        if self.failure == 'missing_terminal':
            return 0
        terminal = dict(before, completed_at=time.time(), exit_code=0, status='completed',
                        container_still_running=False, after_vram='0%', after_kfd_visible=False)
        if self.damage == 'command':
            terminal['command'] = ['other']
        if self.damage == 'release':
            terminal['after_vram'] = '1%'
        if self.damage == 'deadline':
            terminal['completed_at'] = self.deadline['search_stop_at_epoch'] + 1
        if self.damage == 'source_during_device':
            (self.root / 'campaign/engineering/probe/source/entry.py').write_text('changed')
        if self.damage == 'artifact_during_device':
            (self.root / '.local/hip-probes/probe/fixture.bin').write_text('changed')
        if self.damage == 'build_report':
            (self.root / 'campaign/engineering/probe/build.json').write_text('{}')
        if self.damage == 'manifest':
            path = self.root / 'campaign/engineering/probe/manifest.json'
            path.write_text('{}')
        if self.failure == 'numeric':
            terminal['exit_code'] = 1
            terminal['status'] = 'not_qualified'
        write(receipt.with_name(receipt.stem + '-terminal.json'), terminal)
        return 0 if terminal['exit_code'] == 0 else 1

    def run_probe(self, identity='probe', *, kind='bench', skip=''):
        with patch.object(probe, 'binding', return_value=(kind, self.bound)), \
             patch.object(probe.subprocess, 'call', side_effect=self.execute):
            return probe.run_probe(self.root, identity, 'campaign/candidates/request.json', skip)

    def test_one_create_only_probe_uses_canonical_stages_and_never_cake_nomination(self):
        result = self.run_probe()
        self.assertEqual(result['status'], 'measurements_retained_pending_review')
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(self.calls[0][3], self.bound['image'])
        self.assertIn('campaign/hip_probe.py', self.calls[0])
        for args in self.calls[1:]:
            self.assertIn('--latest-start-epoch', args)
            self.assertEqual(args[args.index('--image') + 1], self.bound['image'])
            self.assertIn('HIP_PROBE_BUILD_DIR=/work/.local/hip-probes/probe', args)
        check, caller, timer = self.calls[1:]
        self.assertIn('bwbench.py', check)
        self.assertEqual(check[check.index('--rounds') + 1], '10')
        self.assertEqual(check[check.index('--workloads') + 1], 'all')
        self.assertEqual(check[check.index('--symbol') + 1], 'run')
        self.assertIn('campaign/caller_guards.py', caller)
        self.assertIn('campaign/paired_wall.py', timer)
        self.assertEqual(timer[timer.index('--implementation-kind') + 1], 'native_hip')
        folder = self.root / 'campaign/engineering/probe'
        manifest = probe.read(folder / 'manifest.json')
        self.assertEqual(manifest['entry_symbol'], 'run')
        self.assertEqual(probe.read(folder / 'build.json')['flags'], self.request['build_flags'])
        self.assertEqual((folder / 'gap/000.source').read_bytes(),
                         (self.root / 'campaign/candidates/gap.json').read_bytes())
        self.assertFalse((self.root / 'campaign/evaluations').exists())
        self.assertFalse((folder / 'precision.json').exists())
        self.assertNotIn('conservative_geomean', result)
        self.assertIsNone(development_control.nominee(self.root, self.bound['compiler_commit']))
        write(self.root / 'campaign/intake.json', {'plan': 'campaign/groups/c.json'})
        write(self.root / 'campaign/groups/c.json', {'tasks': []})
        with patch.object(final_confirmation, 'ROOT', self.root), patch('os.chdir'):
            final_confirmation.main()
        self.assertEqual(probe.read(self.root / 'campaign/final-confirmation/result.json')['status'],
                         'missing_qualified_nominee')
        before = {str(p.relative_to(folder)):p.read_bytes() for p in folder.rglob('*') if p.is_file()}
        with self.assertRaises(FileExistsError):
            self.run_probe()
        after = {str(p.relative_to(folder)):p.read_bytes() for p in folder.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_build_failure_and_missing_or_bad_release_never_advance(self):
        variants = [('failure', 'build', 1), ('failure', 'missing_admission', 2),
                    ('failure', 'missing_terminal', 2), ('failure', 'numeric', 2),
                    ('damage', 'command', 2), ('damage', 'release', 2), ('damage', 'deadline', 2)]
        for field, value, count in variants:
            with self.subTest(value=value):
                setattr(self, field, value)
                self.calls = []
                result = self.run_probe(value)
                self.assertEqual(result['status'], 'failed')
                self.assertEqual(len(self.calls), count)
                self.assertTrue((self.root / 'campaign/engineering' / value / 'outcome.json').exists())
                self.assertIn(value, probe.author_context(self.root))
                self.failure = self.damage = None

    def test_development_uses_native_manifest_and_propagates_existing_stop_path(self):
        self.bound.update(compiler=self.bound['compiler_commit'], batch_stop_file=str(self.root / 'STOP-hcu4.json'))
        def execute(argv, **kwargs):
            self.assertEqual(kwargs['env']['BWBENCH_STOP_FILE'], self.bound['batch_stop_file'])
            if 'compare-native' in argv:
                self.calls.append(argv)
                write(self.root / 'campaign/engineering/probe/comparison.json', {'numerically_passed': True})
                return 0
            return self.execute(argv, **kwargs)
        with patch.object(probe, 'binding', return_value=('development', self.bound)), \
             patch.object(probe.subprocess, 'call', side_effect=execute):
            result = probe.run_probe(self.root, 'probe', 'campaign/candidates/request.json')
        self.assertEqual(result['status'], 'measurements_retained_pending_review')
        self.assertEqual(len(self.calls), 3)
        self.assertIn('campaign/development_device.py', self.calls[1])
        self.assertIn('--native-manifest', self.calls[1])
        self.assertNotIn('--emission', self.calls[1])
        self.assertIn('compare-native', self.calls[2])
        self.assertFalse(Path(self.bound['batch_stop_file']).exists())

    def test_unknown_native_release_sets_existing_development_stop(self):
        self.bound.update(compiler=self.bound['compiler_commit'],
                          batch_stop_file=str(self.root / 'STOP-hcu4.json'))
        for damage in ('missing_terminal', 'release', 'command', 'source_and_missing_terminal'):
            with self.subTest(damage=damage):
                self.calls = []
                self.failure = 'missing_terminal' if 'missing_terminal' in damage else None
                self.damage = damage if damage in ('release', 'command') else None
                def execute(argv, **kwargs):
                    code = self.execute(argv, **kwargs)
                    if damage == 'source_and_missing_terminal' and argv[0] == 'python3':
                        path = self.root / argv[argv.index('--receipt') + 1]
                        (path.parent / 'source/entry.py').write_text('changed')
                    return code
                with patch.dict(sys.modules, {'campaign.development_stop': development_stop}), \
                     patch.object(probe, 'binding', return_value=('development', self.bound)), \
                     patch.object(probe.subprocess, 'call', side_effect=execute):
                    first = probe.run_probe(self.root, damage, 'campaign/candidates/request.json')
                    stop = Path(self.bound['batch_stop_file'])
                    self.assertTrue(stop.is_file())
                    saved = stop.read_bytes()
                    self.assertEqual(first['status'], 'failed')
                    count = len(self.calls)
                    again = probe.run_probe(self.root, damage + '-again', 'campaign/candidates/request.json')
                    self.assertEqual(again['status'], 'not_executed')
                    self.assertIn('batch STOP', again['reason'])
                    self.assertEqual(len(self.calls), count)
                    self.assertEqual(stop.read_bytes(), saved)
                    self.assertTrue(development_stop.stopped(self.bound))
                stop.unlink()
                self.failure = self.damage = None

    def test_a_lock_wait_timeout_fails_the_probe_without_stopping_the_card(self):
        self.bound.update(compiler=self.bound['compiler_commit'],
                          batch_stop_file=str(self.root / 'STOP-hcu4.json'))
        self.failure = 'lock_wait'
        with patch.dict(sys.modules, {'campaign.development_stop': development_stop}):
            result = self.run_probe('lock-wait', kind='development')
        self.failure = None
        self.assertEqual(result['status'], 'failed')
        self.assertIn('refused before admission', result['reason'])
        self.assertFalse(Path(self.bound['batch_stop_file']).exists())
        outcome = json.loads((self.root / 'campaign/engineering/lock-wait/outcome.json').read_text())
        self.assertEqual(outcome['admission_receipts'], [])
        from development.development_control import released
        self.assertTrue(released(self.root))

    def test_every_pre_admission_gateway_refusal_fails_the_probe_without_stopping_the_card(self):
        """r6 STOP-hcu2/3 aftermath: an existing owner STOP refused a probe and the missing receipt halted again."""
        from development.development_control import released
        self.bound.update(compiler=self.bound['compiler_commit'],
                          batch_stop_file=str(self.root / 'STOP-hcu4.json'))
        for index, reason in enumerate(development_stop.PRE_ADMISSION_REFUSALS):
            with self.subTest(reason=reason), patch.dict(sys.modules, {'campaign.development_stop': development_stop}):
                self.failure = 'refusal:' + reason
                result = self.run_probe(f'refusal-{index}', kind='development')
                self.failure = None
                self.assertEqual(result['status'], 'failed')
                self.assertIn(reason, result['reason'])
                self.assertFalse(Path(self.bound['batch_stop_file']).exists())
                outcome = json.loads((self.root / f'campaign/engineering/refusal-{index}/outcome.json').read_text())
                self.assertEqual(outcome['admission_receipts'], [])
                self.assertTrue(released(self.root))

    def test_an_unrecognised_failure_without_a_receipt_still_stops_the_card(self):
        self.bound.update(compiler=self.bound['compiler_commit'],
                          batch_stop_file=str(self.root / 'STOP-hcu4.json'))
        self.failure = 'refusal:physical_device_binding_failed: inventory changed'
        with patch.dict(sys.modules, {'campaign.development_stop': development_stop}):
            result = self.run_probe('unrecognised', kind='development')
        self.failure = None
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(Path(self.bound['batch_stop_file']).exists())

    def test_a_lock_wait_string_beside_a_terminal_receipt_still_stops_the_card(self):
        self.bound.update(compiler=self.bound['compiler_commit'],
                          batch_stop_file=str(self.root / 'STOP-hcu4.json'))
        self.failure = 'lock_wait_with_terminal'
        with patch.dict(sys.modules, {'campaign.development_stop': development_stop}):
            result = self.run_probe('lock-wait-terminal', kind='development')
        self.failure = None
        self.assertEqual(result['status'], 'failed')
        self.assertTrue(Path(self.bound['batch_stop_file']).exists())

    def test_released_native_candidate_or_late_failure_does_not_halt_development(self):
        self.bound.update(compiler=self.bound['compiler_commit'],
                          batch_stop_file=str(self.root / 'STOP-hcu4.json'))
        for failure in ('numeric', 'deadline'):
            with self.subTest(failure=failure), \
                 patch.dict(sys.modules, {'campaign.development_stop': development_stop}):
                self.failure = 'numeric' if failure == 'numeric' else None
                self.damage = 'deadline' if failure == 'deadline' else None
                result = self.run_probe(failure, kind='development')
                self.assertEqual(result['status'], 'failed')
                self.assertFalse(Path(self.bound['batch_stop_file']).exists())
                self.failure = self.damage = None

    def test_native_runtime_failure_keeps_existing_development_classification(self):
        self.bound.update(compiler=self.bound['compiler_commit'],
                          batch_stop_file=str(self.root / 'STOP-hcu4.json'))
        self.failure = 'numeric'
        def execute(argv, **kwargs):
            code = self.execute(argv, **kwargs)
            if argv[0] == 'python3':
                kwargs['stdout'].write('HSA_STATUS_ERROR_OUT_OF_RESOURCES\n')
                kwargs['stdout'].flush()
            return code
        with patch.dict(sys.modules, {'campaign.development_stop': development_stop}), \
             patch.object(probe, 'binding', return_value=('development', self.bound)), \
             patch.object(probe.subprocess, 'call', side_effect=execute):
            result = probe.run_probe(self.root, 'runtime', 'campaign/candidates/request.json')
        self.assertEqual(result['status'], 'failed')
        self.assertIn('device runtime or admission service failed',
                      probe.read(Path(self.bound['batch_stop_file']))['reason'])

    def test_prepare_cannot_add_a_frozen_source_member(self):
        def execute(argv, **kwargs):
            code = self.execute(argv, **kwargs)
            if argv[:3] == ['bash', 'scripts/dtk.sh', 'cpu']:
                (self.root / 'campaign/engineering/late/source/late_helper.py').write_text('VALUE = 1\n')
            return code
        with patch.object(probe, 'binding', return_value=('bench', self.bound)), \
             patch.object(probe.subprocess, 'call', side_effect=execute):
            result = probe.run_probe(self.root, 'late', 'campaign/candidates/request.json')
        self.assertEqual(result['status'], 'failed')
        self.assertIn('directory membership changed', result['reason'])
        self.assertEqual(len(self.calls), 1)
        manifest = probe.read(self.root / 'campaign/engineering/late/manifest.json')
        self.assertFalse(any(row['path'].endswith('late_helper.py') for row in manifest['source_files']))

    def test_changed_deadline_cannot_fund_a_later_stage(self):
        def execute(argv, **kwargs):
            result = self.execute(argv, **kwargs)
            changed = dict(self.deadline)
            changed['search_stop_at_epoch'] += 100
            changed['stop_at_epoch'] += 100
            write(self.root / 'campaign/deadline.json', changed)
            return result
        with patch.object(probe, 'binding', return_value=('bench', self.bound)), \
             patch.object(probe.subprocess, 'call', side_effect=execute):
            result = probe.run_probe(self.root, 'probe', 'campaign/candidates/request.json')
        self.assertEqual(result['status'], 'failed')
        self.assertIn('deadline changed', result['reason'])
        self.assertEqual(len(self.calls), 1)

    def test_source_artifact_and_manifest_mutation_stop_the_pipeline(self):
        for damage in ('source_during_prepare', 'source_during_device', 'artifact_during_device', 'manifest', 'build_report'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as other:
                old_root = self.root
                self.root = Path(other).resolve()
                source_fixture(self.root)
                write(self.root / 'campaign/deadline.json', self.deadline)
                self.damage, self.calls = damage, []
                result = self.run_probe()
                self.assertEqual(result['status'], 'failed')
                self.assertIn('changed', result['reason'])
                self.assertLessEqual(len(self.calls), 2)
                self.root = old_root

    def test_expired_stop_and_search_close_retain_unexecuted_evidence(self):
        self.bound['batch_stop_file'] = str(self.root / 'STOP-hcu4.json')
        stop = Path(self.bound['batch_stop_file'])
        stop.write_text('original STOP')
        result = self.run_probe('stopped', kind='development')
        self.assertEqual(result['status'], 'not_executed')
        self.assertIn('batch STOP', result['reason'])
        self.assertEqual(stop.read_text(), 'original STOP')
        stop.unlink()
        request = self.root / 'campaign/SEARCH_CLOSE.json'
        request.write_text('{}')
        result = self.run_probe('closed', kind='development')
        self.assertIn('SEARCH_CLOSE', result['reason'])
        request.unlink()
        write(self.root / 'campaign/deadline.json', dict(started_at_epoch=-8999, search_stop_at_epoch=1, stop_at_epoch=1801))
        result = self.run_probe('expired')
        self.assertIn('cutoff', result['reason'])
        self.assertEqual(self.calls, [])

    def test_skip_and_interrupted_history_are_visible_without_fake_emission(self):
        request = {k:v for k,v in self.request.items() if k not in ('source_dir', 'entry', 'build_flags')}
        write(self.root / 'campaign/candidates/request.json', request)
        result = self.run_probe(skip='Budget cannot fund the full original checks; own deadline retained')
        self.assertEqual(result['status'], 'skipped')
        self.assertEqual(self.calls, [])
        (self.root / 'campaign/engineering/interrupted').mkdir()
        context = probe.author_context(self.root)
        self.assertIn('skipped', context)
        self.assertIn('unknown_incomplete', context)
        self.assertFalse((self.root / 'campaign/engineering/probe/emission.json').exists())

    def test_source_closure_rejects_symlinks_escaping_paths_and_unfixed_entry(self):
        for change in ('directory_symlink', 'file_symlink', 'escape', 'missing_entry', 'non_source'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as other:
                root = Path(other).resolve()
                request = source_fixture(root)
                source = root / request['source_dir']
                if change == 'directory_symlink':
                    (source / 'link').symlink_to(source, target_is_directory=True)
                if change == 'file_symlink':
                    (source / 'link.py').symlink_to(source / 'entry.py')
                if change == 'escape':
                    request['entry'] = '../entry.py'
                if change == 'missing_entry':
                    request['entry'] = 'not_present.py'
                if change == 'non_source':
                    (source / 'arbitrary.json').write_text('{}')
                folder = root / 'campaign/engineering/test'
                folder.mkdir(parents=True)
                with self.assertRaises(ValueError):
                    probe.freeze(root, folder, request, '')
        (self.root / 'campaign/candidates/native/entry.py').write_text('def other(): pass\n')
        result = self.run_probe()
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(len(self.calls), 1)

    def test_binding_reconcile_failure_is_retained_without_build(self):
        with patch.object(probe, 'binding', side_effect=ValueError('frozen source changed')), \
             patch.object(probe.subprocess, 'call') as child:
            result = probe.run_probe(self.root, 'changed', 'campaign/candidates/request.json')
        self.assertEqual(result['status'], 'not_executed')
        self.assertIn('frozen source changed', result['reason'])
        child.assert_not_called()


SOURCE = os.environ.get('BWBENCH_CAKE_SOURCE')


@unittest.skipUnless(SOURCE, 'Set BWBENCH_CAKE_SOURCE to a clean pinned Compiler checkout')
class NativeDevelopmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        (self.root / '.deps').mkdir()
        (self.root / '.deps/cake-ir').symlink_to(Path(SOURCE).resolve(), target_is_directory=True)
        request = source_fixture(self.root)
        for name in ('development_evaluate.py', 'development_device.py'):
            shutil.copyfile(ROOT / 'development' / name, self.root / 'campaign' / name)
        shutil.copyfile(ROOT / 'campaign/hip_probe.py', self.root / 'campaign/hip_probe.py')
        (self.root / 'campaign/development_binding.py').write_text(
            "import json\ndef reconcile(root, mounted=False):\n"
            " return json.loads((root/'campaign/development-binding.json').read_text())\n")
        folder = self.root / 'campaign/engineering/native'
        folder.mkdir(parents=True)
        self.manifest = probe.freeze(self.root, folder, request, '')
        self.prepare_workload()

    def tearDown(self):
        self.temp.cleanup()

    def process(self, source):
        prefix = "import json,sys\nfrom pathlib import Path\nsys.path.insert(0,str(Path('.deps/cake-ir/src').resolve()))\nsys.path.insert(0,str(Path('campaign').resolve()))\n"
        return subprocess.run([sys.executable, '-c', prefix + source], cwd=self.root,
                              text=True, capture_output=True, timeout=60)

    def prepare_workload(self):
        write(self.root / 'campaign/engineering/native/manifest.json', self.manifest)
        result = self.process('''import torch
from open_cake_ir.compiler import Compiler
from open_cake_ir.tasks.workloads import create_task, load_workload, materialize_evaluation_case
from development_device import native_output_contract
from campaign.hip_probe import prepare_native
w,_=create_task('silu',backend='triton-dcu',rows=2,columns=4)
Path('campaign/workload.json').write_text(json.dumps(w))
w=load_workload('campaign/workload.json')
c=Compiler.load('.deps/cake-ir','.deps/cake-ir/compiler/revision.json')
b=dict(task='silu',compiler=c.commit,case_ids=list(w.case_ids))
Path('campaign/development-binding.json').write_text(json.dumps(b))
p=Path('campaign/engineering/native/manifest.json')
m=json.loads(p.read_text());m.update(task='silu',compiler=c.commit);p.write_text(json.dumps(m))
prepare_native(Path.cwd(),'campaign/engineering/native/manifest.json')
prepared=[];observed=[]
for case_id in w.case_ids:
 inputs,expected=materialize_evaluation_case(w,case_id)
 prepared.append(dict(case_id=case_id,flat_inputs=inputs,expected=expected))
 observed.append(dict(case_id=case_id,outputs=expected,after_inputs=inputs,samples_us=[1.0],
  implementation_kind='native_hip',output_metadata=native_output_contract(w,case_id)))
torch.save(prepared,'campaign/development-inputs.pt')
torch.save(observed,'campaign/engineering/native/observed.pt')
''')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_native_comparison_reuses_all_original_cases_and_keeps_no_cake_outcome(self):
        result = subprocess.run([sys.executable, 'campaign/development_evaluate.py', '--phase',
            'compare-native', '--id', 'native', '--native-manifest', 'campaign/engineering/native/manifest.json'],
            cwd=self.root, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = probe.read(self.root / 'campaign/engineering/native/comparison.json')
        self.assertTrue(report['numerically_passed'])
        self.assertEqual(len(report['checks']), 5)
        self.assertNotIn('status', report)
        self.assertFalse((self.root / 'campaign/engineering/native/outcome.json').exists())
        self.assertFalse((self.root / 'campaign/evaluations').exists())
        repeated = subprocess.run([sys.executable, 'campaign/development_evaluate.py', '--phase',
            'compare-native', '--id', 'native', '--native-manifest', 'campaign/engineering/native/manifest.json'],
            cwd=self.root, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(repeated.returncode, 0)
        self.assertIn('FileExistsError', repeated.stderr)

    def test_same_flat_values_with_wrong_output_abi_are_rejected(self):
        result = self.process('''import torch
from open_cake_ir.tasks.workloads import load_workload
from development_device import native_outputs
w=load_workload('campaign/workload.json')
x=torch.ones(2,4)
assert next(iter(native_outputs(w,'primary',x,'cpu')[1].values()))['shape']==[2,4]
for bad in ((x,x),(),x.flatten(),x.half(),x.T.contiguous().T):
 try:native_outputs(w,'primary',bad,'cpu')
 except ValueError:pass
 else:raise AssertionError('bad ABI accepted')
try:native_outputs(w,'primary',x)
except ValueError:pass
else:raise AssertionError('CPU output accepted as assigned logical GPU')
''')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_comparison_rechecks_metadata_and_canonical_numeric_failure(self):
        for damage in ('dtype', 'missing_case', 'value'):
            with self.subTest(damage=damage):
                code = """import torch
p=Path('campaign/engineering/native/observed.pt')
v=torch.load(p,weights_only=False)
"""
                if damage == 'dtype':
                    code += "next(iter(v[0]['output_metadata'].values()))['dtype']='torch.float16'\n"
                elif damage == 'missing_case':
                    code += "v.pop()\n"
                else:
                    code += "v[0]['outputs'][next(iter(v[0]['outputs']))]=[1000.0]*8\n"
                code += "torch.save(v,p)\n"
                self.assertEqual(self.process(code).returncode, 0)
                result = subprocess.run([sys.executable, 'campaign/development_evaluate.py', '--phase',
                    'compare-native', '--id', 'native', '--native-manifest', 'campaign/engineering/native/manifest.json'],
                    cwd=self.root, capture_output=True, text=True, timeout=60)
                self.assertNotEqual(result.returncode, 0)
                result = self.process('''import torch
from open_cake_ir.tasks.workloads import load_workload,materialize_evaluation_case
from development_device import native_output_contract
w=load_workload('campaign/workload.json');v=[]
for cid in w.case_ids:
 inputs,expected=materialize_evaluation_case(w,cid)
 v.append(dict(case_id=cid,outputs=expected,after_inputs=inputs,samples_us=[1.0],
  implementation_kind='native_hip',output_metadata=native_output_contract(w,cid)))
torch.save(v,'campaign/engineering/native/observed.pt')
''')
                self.assertEqual(result.returncode, 0, result.stderr)
                if damage == 'value':
                    self.assertFalse(probe.read(self.root / 'campaign/engineering/native/comparison.json')['numerically_passed'])


if __name__ == '__main__':
    unittest.main()
