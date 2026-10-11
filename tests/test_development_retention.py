"""Whole development CLI path with a real Compiler; no candidate execution or GPU."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = os.environ.get('BWBENCH_CAKE_SOURCE')


@unittest.skipUnless(SOURCE, 'Set BWBENCH_CAKE_SOURCE to a clean pinned Compiler checkout')
class DevelopmentRetentionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.source = Path(SOURCE).resolve()
        (self.root / '.deps').mkdir()
        (self.root / '.deps/cake-ir').symlink_to(self.source, target_is_directory=True)
        (self.root / 'campaign/evaluations').mkdir(parents=True)
        for name in ('development_evaluate.py', 'development_evaluate_owner.py', 'development_stop.py'):
            shutil.copyfile(ROOT / 'development' / name, self.root / 'campaign' / name)
        for name in ('compiler_tools.py', '_compiler_tools.py'):
            shutil.copyfile(ROOT / 'campaign' / name, self.root / 'campaign' / name)
        # Preparation/host binding is the fixture boundary. Parsing, assessment,
        # lowering, workload and the entire evaluator entrypoint are production.
        (self.root / 'campaign/development_binding.py').write_text(
            "import json\ndef reconcile(root, mounted=False):\n"
            " return json.loads((root/'campaign/development-binding.json').read_text())\n")
        self.process('''
from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.tasks.workloads import create_task, load_workload
w, source=create_task('silu',backend='triton-dcu',rows=32,columns=64)
Path('campaign/workload.json').write_text(json.dumps(w))
Path('valid.py').write_text(source)
Path('valid.json').write_text(json.dumps(frontend.parse(source).document))
_, other=create_task('silu',backend='triton-dcu',rows=64,columns=64)
Path('different-abi.py').write_text(other)
loaded=load_workload('campaign/workload.json')
compiler=Compiler.load('.deps/cake-ir','.deps/cake-ir/compiler/revision.json')
Path('campaign/development-binding.json').write_text(json.dumps(dict(
 compiler=compiler.commit,case_ids=list(loaded.case_ids),image='fixture',hcu=4,
 batch_stop_file=str(Path('STOP.json').resolve()))))
Path('campaign/deadline.json').write_text(json.dumps(dict(search_stop_at_epoch=10**12,stop_at_epoch=10**12)))
''')
        self.valid = (self.root / 'valid.py').read_bytes()

    def tearDown(self):
        self.temp.cleanup()

    def process(self, source, *, success=True):
        prefix = "import json,sys\nfrom pathlib import Path\nsys.path.insert(0,str(Path('.deps/cake-ir/src').resolve()))\nsys.path.insert(0,str(Path('campaign').resolve()))\n"
        result = subprocess.run([sys.executable, '-c', prefix + source], cwd=self.root,
                                text=True, capture_output=True, timeout=30)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def evaluate(self, payload=None, *, suffix='.py', identity='attempt', injection='',
                 phase='emit', candidate=None):
        if candidate is None:
            candidate = 'input' + suffix
            (self.root / candidate).write_bytes(payload if payload is not None else self.valid)
        arguments = ['development_evaluate', '--candidate', candidate, '--id', identity, '--phase', phase]
        result = self.process("import runpy\n" + injection + "\nsys.argv=" + repr(arguments) +
                              "\nrunpy.run_path('campaign/development_evaluate.py',run_name='__main__')\n",
                              success=False)
        return result, self.root / 'campaign/evaluations' / identity

    def refusal(self, result, folder, raw, stage):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((folder / 'submitted.source').read_bytes(), raw)
        outcome = json.loads((folder / 'outcome.json').read_text())
        self.assertEqual((outcome['status'], outcome['stage']), ('refused', stage))
        self.assertTrue(outcome['raw_source_complete'])
        self.assertEqual(outcome['submitted_source'], str((folder / 'submitted.source').relative_to(self.root)))
        self.assertFalse((folder / 'emission.json').exists())
        self.assertFalse((folder / 'admission.json').exists())
        return outcome

    def test_parse_refusals_retain_exact_bytes_without_fake_assessment(self):
        for suffix, raw, stage in [('.py', b'def broken(\r\n', 'frontend'),
                                   ('.json', b'{broken\r\n', 'json'),
                                   ('.json', b'[]', 'json'),
                                   ('.py', b'\xff\xfe', 'decode')]:
            with self.subTest(suffix=suffix, stage=stage):
                result, folder = self.evaluate(raw, suffix=suffix, identity='bad-' + stage + str(len(raw)))
                outcome = self.refusal(result, folder, raw, stage)
                self.assertFalse((folder / 'assessment.json').exists())
                self.assertFalse((folder / ('candidate' + suffix)).exists())
                if stage == 'frontend':
                    self.assertIn('location', outcome['error'])
                    self.assertEqual(outcome['error']['location']['filename'], str(self.root / 'input.py'))

    def test_untyped_schema_and_compiler_errors_are_not_candidate_refusals(self):
        for identity, payload, injection in [
            ('schema', b'{}', ''),
            ('internal-assessment', (self.root / 'valid.json').read_bytes(),
             "from open_cake_ir.compiler import Compiler\nfrom open_cake_ir.compiler.errors import CompilerError\ndef fail(*a): raise CompilerError('opaque internal assessment error')\nCompiler.assess=fail\n")]:
            result, folder = self.evaluate(payload, suffix='.json', identity=identity, injection=injection)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((folder / 'submitted.source').read_bytes(), payload)
            outcome = json.loads((folder / 'outcome.json').read_text())
            self.assertEqual((outcome['status'], outcome['stage']), ('evaluation_error', 'assessment'))
            self.assertFalse((folder / 'assessment.json').exists())
            self.assertFalse((folder / 'emission.json').exists())

    def test_task_target_and_abi_refusals_keep_actual_assessment(self):
        for identity, raw, stage in [
            ('target', self.valid.replace(b'gfx938', b'gfx1151'), 'target'),
            ('abi', (self.root / 'different-abi.py').read_bytes(), 'abi')]:
            result, folder = self.evaluate(raw, identity=identity)
            self.refusal(result, folder, raw, stage)
            self.assertTrue(json.loads((folder / 'assessment.json').read_text())['lowering_eligible'])

    def test_verifier_refusal_keeps_the_canonical_assessment(self):
        raw = self.valid.replace(b'gfx938', b'gfx9999')
        result, folder = self.evaluate(raw)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((folder / 'submitted.source').read_bytes(), raw)
        self.assertEqual((folder / 'candidate.py').read_bytes(), raw)
        assessment = json.loads((folder / 'assessment.json').read_text())
        self.assertFalse(assessment['accepted'])
        self.assertIn('TARGET_UNSUPPORTED', [f['code'] for f in assessment['findings']])
        self.assertEqual(json.loads((folder / 'outcome.json').read_text())['status'], 'refused')
        self.assertFalse((folder / 'emission.json').exists())

    def test_io_and_unexpected_lowering_errors_remain_nonzero(self):
        injections = [
            ('source_read', "original=Path.read_bytes\ndef fail(p):\n if p.name=='input.py': raise OSError('fixture read failure')\n return original(p)\nPath.read_bytes=fail\n"),
            ('source_snapshot', "original=Path.open\ndef fail(p,*a,**k):\n if p.name=='submitted.source': raise OSError('fixture write failure')\n return original(p,*a,**k)\nPath.open=fail\n"),
            ('lowering', "from open_cake_ir.compiler import Compiler\ndef fail(*a): raise RuntimeError('fixture lowering failure')\nCompiler.lower=fail\n"),
            ('frontend', "from open_cake_ir.compiler import frontend\ndef fail(*a,**k): raise RuntimeError('fixture parser bug')\nfrontend.parse=fail\n")]
        for stage, injection in injections:
            result, folder = self.evaluate(identity=stage.replace('_', '-'), injection=injection)
            self.assertNotEqual(result.returncode, 0)
            outcome = json.loads((folder / 'outcome.json').read_text())
            self.assertEqual((outcome['status'], outcome['stage']), ('evaluation_error', stage))
            self.assertFalse((folder / 'emission.json').exists())

    def test_snapshot_is_evaluated_after_original_source_changes(self):
        raw = self.valid.replace(b'\n', b'\r\n')
        injection = "original=Path.read_bytes\ndef capture(p):\n value=original(p)\n if p.name=='input.py': p.write_bytes(b'def changed(')\n return value\nPath.read_bytes=capture\n"
        result, folder = self.evaluate(raw, injection=injection)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((folder / 'submitted.source').read_bytes(), raw)
        self.assertEqual((folder / 'candidate.py').read_bytes(), raw)
        self.assertTrue((folder / 'emission.json').exists())
        self.assertEqual((self.root / 'input.py').read_bytes(), b'def changed(')

    def test_json_success_and_duplicate_id_preserve_retained_records(self):
        raw = (self.root / 'valid.json').read_bytes()
        result, folder = self.evaluate(raw, suffix='.json')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((folder / 'candidate.json').read_bytes(), raw)
        before = {p.name:p.read_bytes() for p in folder.iterdir()}
        again, _ = self.evaluate(b'{different', suffix='.json')
        self.assertNotEqual(again.returncode, 0)
        self.assertEqual(before, {p.name:p.read_bytes() for p in folder.iterdir()})

    def test_validate_success_and_failure_never_write_records(self):
        result, folder = self.evaluate()
        self.assertEqual(result.returncode, 0, result.stderr)
        binding = json.loads((self.root / 'campaign/development-binding.json').read_text())
        emission = json.loads((folder / 'emission.json').read_text())
        candidate = str((folder / 'candidate.py').relative_to(self.root))
        # A synthetic prior result exercises validation only, not numerical acceptance.
        (folder / 'outcome.json').write_text(json.dumps(dict(id='attempt',compiler=binding['compiler'],
            status='accepted',phase='search',candidate=candidate,source=emission['source'],
            checks=[dict(case_id=c,passed=True) for c in binding['case_ids']])))
        for damaged in [False, True]:
            if damaged:(folder / 'candidate.py').write_bytes(b'def broken(')
            before = {p.name:p.read_bytes() for p in folder.iterdir()}
            observed, _ = self.evaluate(phase='validate', candidate=candidate)
            self.assertEqual(observed.returncode == 0, not damaged)
            self.assertEqual(before, {p.name:p.read_bytes() for p in folder.iterdir()})

    def test_compare_cannot_replace_an_existing_outcome(self):
        result, folder = self.evaluate()
        self.assertEqual(result.returncode, 0, result.stderr)
        # CPU-only observations/comparator isolate create-only result custody;
        # this synthetic accepted result makes no numerical or timing claim.
        injection = '''import types
import open_cake_ir.evaluation.core as core
core.compare_tile_outputs=lambda *args:(True,{'fixture':True})
cases=json.loads(Path('campaign/development-binding.json').read_text())['case_ids']
prepared=[dict(case_id=c,flat_inputs={},expected={}) for c in cases]
observed=[dict(case_id=c,outputs={},after_inputs={},samples_us=[1.0]) for c in cases]
torch=types.ModuleType('torch')
torch.load=lambda path,**kwargs:prepared if path.name=='development-inputs.pt' else observed
sys.modules['torch']=torch
'''
        result, _ = self.evaluate(phase='compare', candidate='campaign/evaluations/attempt/candidate.py',
                                  injection=injection)
        self.assertEqual(result.returncode, 0, result.stderr)
        before = {p.name:p.read_bytes() for p in folder.iterdir()}
        repeated, _ = self.evaluate(phase='compare', candidate='campaign/evaluations/attempt/candidate.py',
                                    injection=injection)
        self.assertNotEqual(repeated.returncode, 0)
        self.assertIn('FileExistsError', repeated.stderr)
        self.assertEqual(before, {p.name:p.read_bytes() for p in folder.iterdir()})

    def test_refusal_subtree_does_not_weaken_release_audit(self):
        from importlib.util import spec_from_file_location, module_from_spec
        def module(path):
            spec=spec_from_file_location(path.stem,path);value=module_from_spec(spec);spec.loader.exec_module(value);return value
        queue=module(ROOT/'scripts/bench_queue.py')
        control=module(ROOT/'development/development_control.py')
        result, folder = self.evaluate(b'{broken', suffix='.json')
        self.refusal(result, folder, b'{broken', 'json')
        # Audit only the added evaluation evidence. The original bad *.json
        # still fails the whole-Run scanner and is outside this change's scope.
        audit=self.root/'audit';shutil.copytree(folder,audit/'campaign/evaluations/attempt')
        self.assertTrue(control.released(audit))
        self.assertTrue(queue.release_audit(audit)['release_records_clean'])
        bad=audit/'campaign/admission.json';bad.write_text('{')
        self.assertFalse(control.released(audit))
        self.assertFalse(queue.release_audit(audit)['release_records_clean'])

    def test_owner_does_not_advance_refusal_or_error_to_device(self):
        for code in [0, 7]:
            result = self.process('''
import runpy,subprocess
calls=[]
def call(argv,**kwargs):
 calls.append(argv)
 if len(calls)>1: raise AssertionError('unexpected device or compare call')
 return CODE
subprocess.call=call
sys.argv=['development_evaluate_owner','--candidate','valid.py','--id','no-emission']
try:runpy.run_path('campaign/development_evaluate_owner.py',run_name='__main__')
except SystemExit as error:assert error.code==CODE
assert len(calls)==1 and calls[0][2]=='cpu' and '--phase' in calls[0]
'''.replace('CODE',str(code)))
            self.assertEqual(result.returncode,0)


if __name__ == '__main__':
    unittest.main()
