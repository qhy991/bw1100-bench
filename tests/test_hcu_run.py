"""Focused safety checks for the standalone HCU admission boundary."""

import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

MODULE = Path(__file__).resolve().parents[1] / 'scripts/hcu_run.py'
SPEC = importlib.util.spec_from_file_location('hcu_run', str(MODULE))
hcu_run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hcu_run)


class HcuRunTests(unittest.TestCase):
    def test_receipt_cannot_escape_or_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(hcu_run, 'ROOT', Path(temporary)):
                with self.assertRaises(ValueError):
                    hcu_run._receipt_paths('../elsewhere.json')
                path, _ = hcu_run._receipt_paths('campaign/results/job.json')
                path.write_text('{}')
                with self.assertRaises(FileExistsError):
                    hcu_run._receipt_paths('campaign/results/job.json')

    def test_receipt_outside_the_release_walks_is_refused(self):
        """#44: a receipt neither release check reads would certify a killed gateway released."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / 'campaign/real').mkdir(parents=True)
            (root / '.data').mkdir()
            (root / 'campaign/to-data').symlink_to(root / '.data')
            (root / '.local/profile').mkdir(parents=True)
            (root / '.local/profile/to-cache').symlink_to(root / 'campaign/__pycache__')
            (root / 'campaign/to-real').symlink_to(root / 'campaign/real')
            with mock.patch.object(hcu_run, 'ROOT', root):
                for name in ('.data/r/admission.json', 'native/admission.json',
                             'results/rmsnorm-admission.json', 'admission.json', 'campaign', '.',
                             'campaign/__pycache__/admission.json', '.local/triton-cache/a.json',
                             '.local/aiter-jit-cache/x/a.json', 'campaign/.deps/a.json',
                             'campaign/.data/a.json', '.local/.git/a.json',
                             'campaign/admission.txt', 'campaign/to-data/admission.json',
                             '.local/profile/to-cache/admission.json'):
                    with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'release checks'):
                        hcu_run._receipt_paths(name)
                for name, resolved in (('campaign/evaluations/e1/correctness-admission.json', None),
                                       ('.local/profile/RUN/admission.json', None),
                                       ('campaign/to-real/admission.json', 'campaign/real/admission.json')):
                    with self.subTest(name=name):
                        path, terminal = hcu_run._receipt_paths(name)
                        self.assertEqual(path, root / (resolved or name))
                        self.assertEqual(terminal, path.with_name(path.stem + '-terminal.json'))
            self.assertFalse((root / 'campaign/__pycache__').exists())
            self.assertFalse((root / 'native').exists() or (root / 'results').exists())

    def test_out_of_scope_receipt_is_a_caller_error_before_any_admission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stop = root / 'STOP.json'
            argv = ['hcu_run.py', '--image', 'pinned-image', '--device', '0',
                    '--receipt', 'results/job.json', '--', 'python3', 'work.py']
            with mock.patch.object(hcu_run, 'ROOT', root), \
                 mock.patch.object(hcu_run, 'LOCK_DIR', root / '.locks'), \
                 mock.patch.dict(hcu_run.os.environ, {'BWBENCH_STOP_FILE': str(stop)}), \
                 mock.patch.object(sys, 'argv', argv), \
                 mock.patch.object(hcu_run.sys, 'stderr', new_callable=io.StringIO) as stderr, \
                 mock.patch.object(hcu_run.subprocess, 'check_output') as output, \
                 mock.patch.object(hcu_run.subprocess, 'run') as run:
                with self.assertRaises(SystemExit) as refused:
                    hcu_run.main()
                output.assert_not_called()
                run.assert_not_called()
            self.assertEqual(refused.exception.code, 2)
            self.assertIn('under campaign/ or .local/', stderr.getvalue())
            self.assertEqual(sorted(path.name for path in root.iterdir()), [])

    def test_a_terminal_named_receipt_is_refused(self):
        """#45 review: the checks read a *-terminal.json file as a terminal, never as an
        admission awaiting one, so a killed gateway's receipt by that name would pass."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with mock.patch.object(hcu_run, 'ROOT', root):
                for name in ('campaign/job-terminal.json', '.local/profile/a-terminal.json'):
                    with self.subTest(name=name), self.assertRaisesRegex(ValueError, '-terminal.json'):
                        hcu_run._receipt_paths(name)
                path, _ = hcu_run._receipt_paths('campaign/terminal.json')
                self.assertEqual(path, root / 'campaign/terminal.json')

    def test_a_directory_swapped_for_a_symlink_after_the_check_gets_no_receipt(self):
        """#45 review: the lock wait separates the path check from the write. A component
        swapped for a symlink in between must fail the write, not redirect it."""
        for swapped in ('campaign', 'campaign/results', 'campaign/results/e1'):
            for target in ('.data/r', 'outside'):
                with self.subTest(swapped=swapped, target=target), \
                     tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary).resolve() / 'repo'
                    root.mkdir()
                    moved = (root / target) if target != 'outside' else Path(temporary).resolve() / target
                    with mock.patch.object(hcu_run, 'ROOT', root):
                        path, terminal = hcu_run._receipt_paths('campaign/results/e1/job.json')
                        (root / swapped).rename(root / 'moved')
                        moved.mkdir(parents=True)
                        (root / swapped).symlink_to(moved, target_is_directory=True)
                        for name in (path, terminal):
                            with self.assertRaisesRegex(ValueError, 'no longer a plain directory'):
                                hcu_run._write_receipt(name, {'job_id': 'x'})
                    self.assertEqual(list(moved.rglob('*.json')), [])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with mock.patch.object(hcu_run, 'ROOT', root):
                path, _ = hcu_run._receipt_paths('campaign/results/job.json')
                hcu_run._write_receipt(path, {'job_id': 'x'})
                self.assertEqual(json.loads(path.read_text()), {'job_id': 'x'})
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                with self.assertRaises(FileExistsError):
                    hcu_run._write_receipt(path, {'job_id': 'x'})

    def test_main_writes_no_receipt_through_a_directory_swapped_during_the_lock_wait(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / 'repo'
            root.mkdir()
            elsewhere = root / '.data/r'
            argv = ['hcu_run.py', '--image', 'pinned-image', '--device', '0',
                    '--receipt', 'campaign/results/job.json', '--', 'python3', 'work.py']

            def output(command, **_):
                if command[:3] == ['docker', 'image', 'inspect']:
                    # Runs after the path check, before the admission receipt is created.
                    (root / 'campaign/results').rmdir()
                    elsewhere.mkdir(parents=True)
                    (root / 'campaign/results').symlink_to(elsewhere, target_is_directory=True)
                    return 'sha256:pinned\n'
                if command == ['hy-smi']:
                    return 'HCU Temp AvgPwr Perf PwrCap VRAM%\n0 55C 100W auto 800W 0%\n'
                raise AssertionError(command)

            with mock.patch.object(hcu_run, 'ROOT', root), \
                 mock.patch.object(hcu_run, 'LOCK_DIR', root / '.locks'), \
                 mock.patch.object(sys, 'argv', argv), \
                 mock.patch.object(hcu_run.subprocess, 'check_output', side_effect=output), \
                 mock.patch.object(hcu_run.subprocess, 'run',
                                   return_value=subprocess.CompletedProcess([], 1)) as run:
                with self.assertRaisesRegex(ValueError, 'campaign/results is no longer a plain directory'):
                    hcu_run.main()
                self.assertFalse(any(call.args[0][:2] == ['docker', 'run'] for call in run.call_args_list))
            self.assertEqual(list(elsewhere.iterdir()), [])

    def test_busy_hcu_never_starts_container(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            argv = ['hcu_run.py', '--image', 'pinned-image', '--device', '0',
                    '--receipt', 'campaign/results/job.json', '--', 'python3', 'work.py']

            def output(command, **_):
                if command[:3] == ['docker', 'image', 'inspect']:
                    return 'sha256:pinned\n'
                if command == ['hy-smi']:
                    return 'HCU Temp AvgPwr Perf PwrCap VRAM%\n0 55C 100W auto 800W 97%\n'
                raise AssertionError(command)

            with mock.patch.object(hcu_run, 'ROOT', root), \
                 mock.patch.object(hcu_run, 'LOCK_DIR', root / '.locks'), \
                 mock.patch.object(sys, 'argv', argv), \
                 mock.patch.object(hcu_run.subprocess, 'check_output', side_effect=output), \
                 mock.patch.object(hcu_run.subprocess, 'run') as run:
                with self.assertRaisesRegex(RuntimeError, 'not idle'):
                    hcu_run.main()
                run.assert_called_once_with(['fuser', '/dev/kfd'],
                                            stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL, timeout=10)
            self.assertFalse((root / 'campaign/results/job.json').exists())

    def test_zero_exit_with_busy_post_state_is_not_qualified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            argv = ['hcu_run.py', '--image', 'pinned-image', '--device', '0',
                    '--receipt', 'campaign/results/job.json', '--timeout', '20', '--', 'python3', 'work.py']
            vram = iter(('0%', '97%'))

            def output(command, **_):
                if command[:3] == ['docker', 'image', 'inspect']:
                    return 'sha256:pinned\n'
                if command == ['hy-smi']:
                    return 'HCU Temp AvgPwr Perf PwrCap VRAM%%\n0 55C 100W auto 800W %s\n' % next(vram)
                if command[:3] == ['docker', 'ps', '-q']:
                    return ''
                raise AssertionError(command)

            def run(command, **_):
                if command[0] == 'fuser':
                    return subprocess.CompletedProcess(command, 1)
                if command[:2] == ['docker', 'run']:
                    self.assertIn('sha256:pinned', command)
                    self.assertNotIn('pinned-image', command)
                    self.assertIn('HOME=/tmp', command)
                    self.assertIn('14', command)
                    self.assertTrue(any('/usr/bin/timeout' in part for part in command))
                    return subprocess.CompletedProcess(command, 0)
                raise AssertionError(command)

            with mock.patch.object(hcu_run, 'ROOT', root), \
                 mock.patch.object(hcu_run, 'LOCK_DIR', root / '.locks'), \
                 mock.patch.object(sys, 'argv', argv), \
                 mock.patch.object(hcu_run.subprocess, 'check_output', side_effect=output), \
                 mock.patch.object(hcu_run.subprocess, 'run', side_effect=run), \
                 mock.patch.object(hcu_run.time, 'sleep'), \
                 mock.patch.object(hcu_run.time, 'monotonic', side_effect=(0, 31)):
                self.assertEqual(hcu_run.main(), 75)
            terminal = json.loads((root / 'campaign/results/job-terminal.json').read_text())
            self.assertEqual(terminal['status'], 'not_qualified')
            self.assertEqual(terminal['after_vram'], '97%')
            self.assertFalse(terminal['physical_exclusivity'])
            self.assertEqual(terminal['container_timeout_s'], 14)

    def test_success_waits_for_observed_delayed_release(self):
        with mock.patch.object(hcu_run, '_vram', side_effect=('32%', '0%')), \
             mock.patch.object(hcu_run, '_kfd_visible', return_value=False), \
             mock.patch.object(hcu_run.subprocess, 'check_output', return_value=''), \
             mock.patch.object(hcu_run.time, 'sleep'), \
             mock.patch.object(hcu_run.time, 'monotonic', side_effect=(0, 1, 2)):
            vram, kfd, live, observations = hcu_run._observe_release(2, 'owned-job', 0)
        self.assertEqual((vram, kfd, live), ('0%', False, False))
        self.assertEqual([row['vram'] for row in observations], ['32%', '0%'])


if __name__ == '__main__':
    unittest.main()
