"""Real pinned-Compiler CPU integration; no model, native compilation or GPU."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

from campaign import compiler_tools as tools

ROOT = Path(__file__).resolve().parents[1]
SOURCE = os.environ.get('BWBENCH_CAKE_SOURCE')


class ToolBoundaryTests(unittest.TestCase):
    def test_foreign_parent_and_invalid_action_paths_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for parent in ('../foreign.py', '/etc/passwd', '.deps/cake-ir/examples/python/fma.py'):
                with self.subTest(parent=parent), self.assertRaises(ValueError):
                    tools.parent_source(root, parent)
            (root / 'campaign/candidates').mkdir(parents=True)
            (root / 'campaign/candidates/link.py').symlink_to('/etc/passwd')
            with self.assertRaises(ValueError): tools.parent_source(root, 'campaign/candidates/link.py')
            with self.assertRaises(ValueError): tools.action_directory(root, '../reuse')

    def test_incomplete_and_unreadable_actions_remain_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('partial', 'unreadable'):
                folder = root / tools.ACTIONS / name
                folder.mkdir(parents=True)
                (folder / 'request.json').write_text(json.dumps(dict(started_at_epoch=1, action={})))
            (root / tools.ACTIONS / 'unreadable/result.json').write_text('{')
            rows = tools.history(root)['actions']
            self.assertEqual({row['reason'] for row in rows}, {'unknown_incomplete', 'unknown_unreadable'})


@unittest.skipUnless(SOURCE, 'Set BWBENCH_CAKE_SOURCE to a clean pinned successor checkout for real Compiler CPU checks')
class RealCompilerToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.source = Path(SOURCE).resolve()
        self.pin = subprocess.check_output(['git', '-C', str(self.source), 'rev-parse', 'HEAD'], text=True).strip()
        (self.root / '.deps').mkdir()
        (self.root / '.deps/cake-ir').symlink_to(self.source, target_is_directory=True)
        (self.root / 'campaign/candidates').mkdir(parents=True)
        for name in ('compiler_tools.py', '_compiler_tools.py'):
            shutil.copyfile(ROOT / 'campaign' / name, self.root / 'campaign' / name)
        shutil.copyfile(ROOT / 'campaign/cake_bridge.py', self.root / 'campaign/cake_bridge.py')
        (self.root / 'campaign/groups').mkdir()
        self.write('campaign/groups/c.json', dict(compiler_commit=self.pin))
        self.write('campaign/binding.json', dict(compiler_commit=self.pin))
        (self.root / '.gitignore').write_text('.deps/\n__pycache__/\ncampaign/candidates/\ncampaign/compiler-actions/\ncampaign/deadline.json\ncampaign/intake.json\n')
        self.git('init', '-q'); self.git('config', 'user.name', 'Tool fixture')
        self.git('config', 'user.email', 'fixture@example.invalid')
        self.tool('freeze')
        self.git('add', '.'); self.git('commit', '-qm', 'Frozen fixture')
        self.prepared = self.git('rev-parse', 'HEAD')
        self.write('campaign/intake.json', dict(source_commit=self.prepared))
        self.write('campaign/deadline.json', dict(search_stop_at_epoch=time.time() + 1200))
        self.process('''
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path('.deps/cake-ir/src').resolve()))
from open_cake_ir.tasks.workloads import create_task
for task in ('gemm','silu'):
    workload, source=create_task(task,backend='triton-dcu',rows=32,columns=64,depth=128 if task=='gemm' else None)
    Path('campaign/candidates/'+task+'.py').write_text(source)
    Path('campaign/candidates/'+task+'-workload.json').write_text(json.dumps(workload))
''')

    def tearDown(self):
        self.temp.cleanup()

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.root), *args], text=True).strip()

    def write(self, path, value):
        (self.root / path).write_text(json.dumps(value))

    def process(self, source):
        result = subprocess.run([sys.executable, '-c', source], cwd=self.root,
                                text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def tool(self, *args, success=True):
        result = subprocess.run([sys.executable, 'campaign/compiler_tools.py', *args],
                                cwd=self.root, text=True, capture_output=True, timeout=60)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return json.loads(result.stdout)
        self.assertNotEqual(result.returncode, 0)
        return result

    def request(self, parent='gemm'):
        source = 'campaign/candidates/' + parent + '.py'
        stage = self.tool('inspect', '--parent', source)['stages'][0]['name']
        return dict(action='transform', parent=source, transformation='specialize_fp32_contraction',
                    parameters=dict(stage=stage, row_tile=16, column_tile=32, k_tile=32,
                                    num_warps=4, num_stages=1, schedule_id='mma_trial', entry_point='mma_trial'))

    def apply(self, identity, request):
        self.write('campaign/candidates/request.json', request)
        return self.tool('transform', '--id', identity, '--request', 'campaign/candidates/request.json')

    def test_real_mma_candidate_emits_and_replays_without_device_work(self):
        self.assertEqual(self.apply('mma', self.request())['reason'], 'applied')
        self.assertEqual(self.tool('verify', '--id', 'mma')['replay'], 'matched')
        result = subprocess.run([sys.executable, 'campaign/cake_bridge.py', 'emit',
            'campaign/compiler-actions/mma/stage-000.json', '--output-dir', 'campaign/compiler-actions/mma/emission'],
            cwd=self.root, text=True, capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        generated = (self.root / 'campaign/compiler-actions/mma/emission/source.py').read_text()
        self.assertIn('tl.dot', generated)
        self.assertFalse((self.root / 'campaign/DONE.json').exists())
        # The original author source may evolve; the retained action owns its snapshot.
        (self.root / 'campaign/candidates/gemm.py').write_text('# next proposal')
        self.assertEqual(self.tool('verify', '--id', 'mma')['replay'], 'matched')
        self.write('campaign/compiler-actions/mma/stage-000.json', {})
        self.tool('verify', '--id', 'mma', success=False)

    def test_refusal_unknown_transform_and_duplicate_are_retained(self):
        refused = self.apply('not-mma', self.request('silu'))
        self.assertNotEqual(refused['reason'], 'applied')
        self.assertFalse((self.root / 'campaign/compiler-actions/not-mma/program.json').exists())
        self.assertEqual(self.tool('verify', '--id', 'not-mma')['replay'], 'matched')
        request = self.request(); request['transformation'] = 'not_declared'
        self.assertEqual(self.apply('unknown', request)['reason'], 'transform_not_granted')
        self.assertEqual(self.tool('verify', '--id', 'unknown')['replay'], 'matched')
        record = (self.root / 'campaign/compiler-actions/unknown/result.json').read_bytes()
        self.tool('transform', '--id', 'unknown', '--request', 'campaign/candidates/request.json', success=False)
        self.assertEqual((self.root / 'campaign/compiler-actions/unknown/result.json').read_bytes(), record)
        request = json.loads((self.root / 'campaign/compiler-actions/unknown/request.json').read_text())
        request['action']['transformation'] = 'specialize_fp32_contraction'
        self.write('campaign/compiler-actions/unknown/request.json', request)
        self.tool('verify', '--id', 'unknown', success=False)

    def test_catalog_drift_and_closed_search_fail_before_new_action(self):
        request = self.request(); self.write('campaign/candidates/request.json', request)
        self.write('campaign/deadline.json', dict(search_stop_at_epoch=0))
        self.tool('transform', '--id', 'late', '--request', 'campaign/candidates/request.json', success=False)
        self.assertFalse((self.root / 'campaign/compiler-actions').exists())
        original = (self.root / tools.CATALOG).read_bytes()
        self.write(tools.CATALOG, dict(compiler_commit=self.pin, transformations=[]))
        self.tool('catalog', success=False)
        (self.root / tools.CATALOG).write_bytes(original)
        self.write('campaign/binding.json', dict(compiler_commit='0' * 40))
        self.tool('catalog', success=False)

    def test_approved_early_closure_refuses_new_actions_but_keeps_replay(self):
        self.apply('before-close', self.request())
        (self.root / 'campaign/binding.json').unlink()
        self.write('campaign/development-binding.json', dict(compiler=self.pin))
        self.write('campaign/deadline.json', dict(source_commit=self.prepared,
                   search_stop_at_epoch=time.time() + 1200))
        self.write('campaign/search-close-owner.json', dict(decision='approved'))
        self.tool('transform', '--id', 'after-close', '--request', 'campaign/candidates/request.json', success=False)
        self.assertFalse((self.root / 'campaign/compiler-actions/after-close').exists())
        self.assertEqual(self.tool('verify', '--id', 'before-close')['replay'], 'matched')

    def test_development_json_emission_keeps_transform_origin_through_validation(self):
        self.apply('mma', self.request())
        shutil.copyfile(ROOT / 'development/development_evaluate.py', self.root / 'campaign/development_evaluate.py')
        shutil.copyfile(self.root / 'campaign/candidates/gemm-workload.json', self.root / 'campaign/workload.json')
        (self.root / 'campaign/evaluations').mkdir()
        self.process('''
import json,runpy,sys,types
from pathlib import Path
sys.path.insert(0,str(Path('.deps/cake-ir/src').resolve()))
from open_cake_ir.tasks.workloads import load_workload
w=load_workload('campaign/workload.json')
binding=dict(compiler=json.loads(Path('campaign/binding.json').read_text())['compiler_commit'],case_ids=list(w.case_ids))
module=types.ModuleType('development_binding');module.reconcile=lambda root,mounted=False:binding
sys.modules['development_binding']=module
sys.argv=['development_evaluate','--candidate','campaign/compiler-actions/mma/stage-000.json','--id','evaluation','--phase','emit']
runpy.run_path('campaign/development_evaluate.py',run_name='__main__')
folder=Path('campaign/evaluations/evaluation')
assert (folder/'candidate.json').exists() and not (folder/'candidate.py').exists()
emission=json.loads((folder/'emission.json').read_text())
assert emission['author_action_stage']=='campaign/compiler-actions/mma/stage-000.json'
# Fixture-only prior outcome exercises the retained-source validator; no numerical result is claimed.
(folder/'outcome.json').write_text(json.dumps(dict(id='evaluation',compiler=binding['compiler'],status='accepted',phase='search',candidate=str(folder/'candidate.json'),source=str(folder/'source.py'),checks=[dict(case_id=case,passed=True) for case in w.case_ids])))
sys.argv=['development_evaluate','--candidate',str(folder/'candidate.json'),'--id','evaluation','--phase','validate']
runpy.run_path('campaign/development_evaluate.py',run_name='__main__')
Path('campaign/compiler-actions/mma/stage-000.json').write_text('{}')
try:
    runpy.run_path('campaign/development_evaluate.py',run_name='__main__')
except ValueError:
    pass
else:
    raise AssertionError('changed tool result was accepted')
''')

    @unittest.skipUnless(os.environ.get('BWBENCH_CONTROL_SOURCE'), 'Optional clean pre-feature Compiler checkout')
    def test_control_catalog_does_not_advertise_successor_only_pass(self):
        old = self.root / 'old-condition'
        (old / 'campaign').mkdir(parents=True); (old / '.deps').mkdir()
        source = Path(os.environ['BWBENCH_CONTROL_SOURCE']).resolve()
        pin = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        (old / '.deps/cake-ir').symlink_to(source, target_is_directory=True)
        for name in ('compiler_tools.py', '_compiler_tools.py'):
            shutil.copyfile(ROOT / 'campaign' / name, old / 'campaign' / name)
        (old / 'campaign/binding.json').write_text(json.dumps(dict(compiler_commit=pin)))
        result = subprocess.run([sys.executable, 'campaign/compiler_tools.py', 'freeze'], cwd=old,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        before = json.loads(result.stdout)
        after = self.tool('catalog')
        self.assertNotIn('specialize_fp32_contraction', [x['name'] for x in before['transformations']])
        self.assertIn('specialize_fp32_contraction', [x['name'] for x in after['transformations']])
        self.assertEqual(before['compiler_commit'], pin)

    def test_both_real_flow_functions_deliver_catalog_to_agent(self):
        for development in (False, True):
            with self.subTest(development=development):
                flow = ROOT / ('development/ralph_flow.py' if development else 'campaign/ralph_flow.py')
                shutil.copyfile(flow, self.root / 'campaign/ralph_flow.py')
                if development:
                    (self.root / 'campaign/binding.json').unlink()
                    self.write('campaign/development-binding.json', dict(compiler=self.pin,
                               batch_stop_file=str(self.root / 'never-stop')))
                    shutil.copyfile(ROOT / 'development/development_stop.py', self.root / 'campaign/development_stop.py')
                else:
                    shutil.copyfile(ROOT / 'campaign/statecard.py', self.root / 'campaign/statecard.py')
                    (self.root / 'campaign/evaluations').mkdir(); (self.root / 'campaign/management').mkdir()
                self.write('campaign/deadline.json', dict(source_commit=self.prepared, search_stop_at_epoch=time.time()+60))
                observed = json.loads(self.process('''
import json,runpy,sys,time,types
from pathlib import Path
package=types.ModuleType('hmz'); flows=types.ModuleType('hmz.flows')
flows.Agent=object; flows.Allowance=lambda **kw:None; flows.flow=lambda **kw:lambda f:f
sys.modules['hmz']=package;sys.modules['hmz.flows']=flows
time.sleep=lambda seconds:None
module=runpy.run_path('campaign/ralph_flow.py'); calls=[]
def author(prompt,**kw):
    calls.append(prompt);path=Path('campaign/deadline.json');d=json.loads(path.read_text());d['search_stop_at_epoch']=0;path.write_text(json.dumps(d))
module['run']((author,), 'Frozen task')
print(json.dumps(calls))
'''))
                self.assertEqual(len(observed), 1)
                self.assertIn(self.pin, observed[0])
                self.assertIn('specialize_fp32_contraction', observed[0])
                self.assertIn('row_tile', observed[0])
                self.assertIn('Own transform observations', observed[0])


if __name__ == '__main__':
    unittest.main()
