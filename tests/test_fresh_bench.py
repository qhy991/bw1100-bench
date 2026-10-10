"""CPU contracts for frozen allocation custody and terminal failure handling."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from campaign.binding import reconcile, validate_plan, validate_routing
from campaign.completion import confirmed_row
from campaign.precision_guard import validate as precision
from campaign import evaluate


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


prepare = module('prepare_bench')
queue = module('bench_queue')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def plan(suite, bench='b' * 40):
    tasks = [row['id'] for row in suite['tasks']]
    return {'kind': 'rolling_bench_fresh_search', 'target': 'gfx938', 'task_ids': tasks,
            'replicates': 1, 'bindings': {'bench': {'commit': bench},
                'control': {'commit': 'c' * 40}, 'successor': {'commit': 'd' * 40}},
            'author': {'model': 'claude/glm-5.3:high', 'scaffold': 'bw1100-bench@' + bench + ':campaign/TASK.md',
                       'knowledge': 'none', 'reference_access': 'known_kernel_reproduction',
                       'wall_time_seconds': 10800, 'confirmation_seconds': 1800},
            'measurement_protocol': 'campaign/protocol.json',
            'total_author_wall_seconds': 24 * 10800,
            'baseline_files': {task: 'baselines/' + str(i) + '.py' for i, task in enumerate(tasks)},
            'allocations': [{'task': task, 'condition': condition, 'replicate': 1,
                             'workspace': '/intent/runs/t%03d-r1-%s' % (i + 1, condition)}
                            for i, task in enumerate(tasks) for condition in ('control', 'successor')]}


def routing(frozen, parent):
    return {'hosts': {'local': {}, 'other': {}}, 'assignments': [
        {'workspace': row['workspace'], 'host': 'local' if index < 2 else 'other',
         'root': str(parent / Path(row['workspace']).name), 'hcu': 1 if index < 2 else 2}
        for index, row in enumerate(frozen['allocations'])]}


class FreshBenchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.suite = json.loads((ROOT / 'suite.json').read_text())
        self.plan = plan(self.suite)

    def tearDown(self):
        self.temp.cleanup()

    def test_all_24_assignments_and_route_pairing(self):
        validate_plan(self.plan, self.suite)
        routes = routing(self.plan, self.root)
        validate_routing(self.plan, routes)
        for mutation in ('missing', 'duplicate', 'foreign', 'cross_host', 'cross_hcu', 'same_root'):
            bad = copy.deepcopy(routes)
            if mutation == 'missing': bad['assignments'].pop()
            if mutation == 'duplicate': bad['assignments'][-1] = bad['assignments'][0]
            if mutation == 'foreign': bad['assignments'][0]['workspace'] = '/not-assigned'
            if mutation == 'cross_host': bad['assignments'][1]['host'] = 'other'
            if mutation == 'cross_hcu': bad['assignments'][1]['hcu'] = 2
            if mutation == 'same_root': bad['assignments'][1]['root'] = bad['assignments'][0]['root']
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_routing(self.plan, bad)
        bad = copy.deepcopy(self.plan)
        bad['allocations'][-1] = bad['allocations'][0]
        with self.assertRaises(ValueError): validate_plan(bad, self.suite)
        bad = copy.deepcopy(self.plan)
        bad['task_ids'].pop()
        with self.assertRaises(ValueError): validate_plan(bad, self.suite)

    def test_existing_queue_reservation_never_restarts(self):
        state = self.root / 'owner'
        state.mkdir()
        write(state / 'LAUNCH.json', {'owner_pid': 1})
        with patch.object(queue.subprocess, 'Popen') as spawn, self.assertRaises(AssertionError):
            queue.run(state, self.root / 'absent-allocation.json')
        spawn.assert_not_called()

    def test_release_unknown_is_retained(self):
        admission = {'schema': 'bw1100-bench.hcu-admission.v1', 'job_id': 'owned'}
        write(self.root / 'a.json', admission)
        self.assertEqual(queue.release_audit(self.root)['issues'][0]['reason'], 'missing_terminal')
        write(self.root / 'a-terminal.json', {**admission, 'completed_at': 1, 'exit_code': 0,
              'container_still_running': False, 'after_vram': '0%', 'after_kfd_visible': False})
        self.assertTrue(queue.release_audit(self.root)['release_records_clean'])
        (self.root / 'unknown.json').write_text('{')
        self.assertFalse(queue.release_audit(self.root)['release_records_clean'])
        self.assertEqual(queue.done_observation(self.root / 'DONE.json'), (None, 'missing_DONE'))

    def test_new_gemm_precision(self):
        for task, dtype in [('L1/003_lm_head_projection_with_logit_slicing', 'bf16'),
                            ('L1/077_whisper_decoder_output_projection', 'fp16')]:
            schedule = {'buffers': [{'name': 'a', 'dtype': dtype}, {'name': 'b', 'dtype': dtype}],
                        'operations': [{'kind': 'mma', 'reads': ['a', 'b'], 'parameters': {'accumulator': 'fp32'}}]}
            write(self.root / 'schedule.json', schedule)
            write(self.root / 'receipt.json', {'schedule': 'schedule.json'})
            self.assertEqual(precision(self.root, task, ['receipt.json'])['status'], 'passed')
            schedule['operations'][0]['parameters']['accumulator'] = dtype
            write(self.root / 'schedule.json', schedule)
            with self.assertRaises(ValueError): precision(self.root, task, ['receipt.json'])

    def test_pre_gpu_rejection_has_outcome_and_cannot_reuse_id(self):
        write(self.root / 'campaign/intake.json', {'plan': 'campaign/groups/c.json', 'image': 'test'})
        write(self.root / 'campaign/groups/c.json', {'arm': 'cake_ir', 'hcu': 1, 'tasks': [{'id': 'task'}]})
        write(self.root / 'campaign/deadline.json', {'search_stop_at_epoch': 0})
        (self.root / 'campaign/candidates').mkdir()
        (self.root / 'candidate.py').write_text('def run(): pass\n')
        argv = ['evaluate', '--candidate', 'candidate.py', '--id', 'bad', '--profile-skip-reason', 'CPU refusal before device']
        old_cwd = Path.cwd()
        try:
            with patch.object(evaluate, 'ROOT', self.root), patch.object(sys, 'argv', argv), patch.object(evaluate.subprocess, 'call') as device:
                self.assertEqual(evaluate.main(), 1)
                device.assert_not_called()
                result = json.loads((self.root / 'campaign/evaluations/bad/outcome.json').read_text())
                self.assertIn('declared emission receipts', result['reason'])
                self.assertEqual(result['admission_receipts'], [])
                with self.assertRaises(FileExistsError): evaluate.main()
        finally:
            os.chdir(old_cwd)

    def test_failed_confirmation_does_not_promote_search_winner(self):
        c = self.root / 'campaign'
        c.mkdir()
        (c / 'finish.py').write_bytes((ROOT / 'campaign/finish.py').read_bytes())
        (c / 'completion.py').write_bytes((ROOT / 'campaign/completion.py').read_bytes())
        (c / 'candidate.py').write_text('def run(): pass\n')
        digest = hashlib.sha256((c / 'candidate.py').read_bytes()).hexdigest()
        write(c / 'intake.json', {'plan': 'campaign/group.json', 'protocol': 'campaign/protocol.json',
            'condition': 'successor', 'replicate': 1, 'compiler_commit': 'pinned',
            'frozen_workspace': '/intent/one', 'experiment_kind': 'rolling_bench_fresh_search'})
        write(c / 'group.json', {'arm': 'cake_ir', 'tasks': [{'id': 'task'}]})
        write(c / 'protocol.json', {'checkpoint_minutes': [1], 'budget_hours': 3})
        write(c / 'deadline.json', {'started_at_epoch': 0, 'search_stop_at_epoch': 10, 'stop_at_epoch': 20})
        write(c / 'evaluations/winner/outcome.json', {'id': 'winner', 'status': 'accepted',
            'completed_at_epoch': 5, 'candidate_source': 'campaign/candidate.py',
            'source_sha256': digest, 'conservative_geomean': 2})
        (c / 'evaluations/unknown').mkdir()
        write(c / 'final-confirmation/result.json', {'status': 'confirmation_failed'})
        subprocess.run([sys.executable, str(c / 'finish.py')], check=True, stdout=subprocess.DEVNULL)
        self.assertEqual(json.loads((c / 'DONE.json').read_text())['tasks'][0]['status'], 'blocked')
        result = json.loads((c / 'ENDPOINT.json').read_text())
        self.assertEqual(result['best_speedup'], 2)
        self.assertEqual(len(result['unknown_attempts']), 1)
        self.assertEqual(result['final_confirmation']['status'], 'confirmation_failed')

    def test_confirmation_cannot_switch_to_another_search_candidate(self):
        candidates = []
        for name in ('confirmed', 'other'):
            path = self.root / (name + '.py')
            path.write_text('def run(): return ' + repr(name))
            candidates.append({'id': name, 'candidate_source': path.name,
                'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'handoff_row': {'task': 'task', 'candidate_source': path.name, 'reason': 'search passed'}})
        write(self.root / 'campaign/final-confirmation/nomination.json', candidates[0])
        confirmation = {'status': 'confirmed', 'nominated_id': 'confirmed'}
        with self.assertRaisesRegex(ValueError, 'missing, stale'):
            confirmed_row(self.root, confirmation, candidates[1:], {})
        with patch('campaign.completion.validate') as gate:
            result = confirmed_row(self.root, confirmation, candidates, {})
            self.assertEqual(result['candidate_source'], 'confirmed.py')
            self.assertEqual(gate.call_args.args[1]['tasks'][0]['latency_report'], 'campaign/final-confirmation/latency.json')
        with patch('campaign.completion.validate', side_effect=ValueError('timing binds other source')):
            with self.assertRaisesRegex(ValueError, 'timing binds other source'):
                confirmed_row(self.root, confirmation, candidates, {})
        (self.root / 'confirmed.py').write_text('# changed after confirmation')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            confirmed_row(self.root, confirmation, candidates, {})

    def test_prepare_real_git_pair_is_fresh_and_reconciles(self):
        def git(repo, *args):
            return subprocess.check_output(['git', '-C', str(repo), *args], text=True, stderr=subprocess.DEVNULL).strip()
        def repo(name):
            path = self.root / name
            path.mkdir()
            git(path, 'init', '-q')
            git(path, 'config', 'user.name', 'Bench Test')
            git(path, 'config', 'user.email', 'bench-test@example.invalid')
            return path
        def commit(path):
            git(path, 'add', '.')
            git(path, 'commit', '-qm', 'CPU fixture')
            return git(path, 'rev-parse', 'HEAD')
        sol = repo('sol')
        (sol / 'source').write_text('pinned fixture')
        sol_pin = commit(sol)
        compilers = {}
        for condition in ('control', 'successor'):
            source = repo(condition)
            write(source / 'compiler/revision.json', {'fixture': condition})
            compilers[condition] = (source, commit(source))
        bench = repo('bench')
        write(bench / 'suite.json', self.suite)
        write(bench / 'sources.lock.json', {'sol_execbench': {'revision': sol_pin}})
        (bench / '.gitignore').write_text('.deps/\n.data/\n.local/\n')
        for name in ('TASK.md', 'protocol.json'):
            target = bench / 'campaign' / name
            target.parent.mkdir(exist_ok=True)
            target.write_bytes((ROOT / 'campaign' / name).read_bytes())
        for name in prepare.GATEWAY_FILES:
            target = bench / name
            target.parent.mkdir(exist_ok=True)
            target.write_text('qualified fixture route\n')
        frozen = plan(self.suite)
        data = self.root / 'data'
        for task, name in frozen['baseline_files'].items():
            target = bench / name
            target.parent.mkdir(exist_ok=True)
            target.write_text('baseline fixture\n')
            folder = data / 'benchmark' / task
            folder.mkdir(parents=True)
            write(folder / 'definition.json', {})
            (folder / 'reference.py').write_text('# original fixture\n')
            (folder / 'workload.jsonl').write_text(''.join(json.dumps({'uuid': str(i)}) + '\n' for i in range(16)))
        bench_pin = commit(bench)
        frozen['bindings']['bench']['commit'] = bench_pin
        frozen['author']['scaffold'] = 'bw1100-bench@' + bench_pin + ':campaign/TASK.md'
        for condition, (_, pin) in compilers.items(): frozen['bindings'][condition]['commit'] = pin
        routes = routing(frozen, self.root / 'physical-runs')
        flags = self.root / 'flags'
        flags.mkdir()
        (flags / 'source').write_text('qualified fixture')
        routes['hosts']['local'] = {'gateway': {'root': str(bench), 'commit': bench_pin},
            'image': 'sha256:' + 'f' * 64, 'home': str(self.root), 'capacity_per_hcu': 1,
            'start_window_seconds': 100, 'compiler_roots': {key: str(value[0]) for key, value in compilers.items()},
            'sol_root': str(sol), 'data_root': str(data), 'flag_gems_root': str(flags)}
        write(self.root / 'plan.json', frozen)
        write(self.root / 'routing.json', routes)
        env = {'GIT_AUTHOR_NAME': 'Bench Test', 'GIT_AUTHOR_EMAIL': 'bench-test@example.invalid',
               'GIT_COMMITTER_NAME': 'Bench Test', 'GIT_COMMITTER_EMAIL': 'bench-test@example.invalid'}
        with patch.object(prepare, 'ROOT', bench), patch.dict(os.environ, env):
            rows = prepare.prepare(self.root / 'plan.json', self.root / 'routing.json', 'local', self.root / 'allocation.json')
            self.assertEqual(len(rows), 2)
            for row in rows:
                run = Path(row['root'])
                binding = reconcile(run)
                self.assertEqual(binding['frozen_workspace'], row['frozen_workspace'])
                self.assertNotEqual(binding['frozen_workspace'], row['root'])
                self.assertEqual(list((run / 'campaign/candidates').iterdir()), [])
                self.assertFalse((run / 'campaign/intake.json').exists())
                self.assertFalse((run / 'campaign/prior').exists())
                self.assertEqual(git(run, 'ls-files', '.data'), '')
                self.assertEqual(git(run, 'status', '--porcelain'), '')
                self.assertEqual((run / 'scripts/hcu_device_identity.py').read_text(), 'qualified fixture route\n')
                self.assertEqual(binding['gateway_files'], list(prepare.GATEWAY_FILES))
            with self.assertRaises(FileExistsError):
                prepare.prepare(self.root / 'plan.json', self.root / 'routing.json', 'local', self.root / 'allocation.json')
            git(bench, 'rm', 'scripts/hcu_device_identity.py')
            git(bench, 'commit', '-qm', 'Missing helper negative fixture')
            with self.assertRaisesRegex(ValueError, 'hcu_device_identity.py'):
                prepare.gateway_sources(bench, git(bench, 'rev-parse', 'HEAD'))
            run = Path(rows[0]['root'])
            modified = json.loads((run / 'campaign/groups/c.json').read_text())
            modified['hcu'] = 2
            write(run / 'campaign/groups/c.json', modified)
            with self.assertRaises(ValueError): reconcile(run)


if __name__ == '__main__':
    unittest.main()
