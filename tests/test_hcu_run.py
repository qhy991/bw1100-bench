"""Focused safety checks for the standalone HCU admission boundary."""

import importlib.util
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
                path, _ = hcu_run._receipt_paths('results/job.json')
                path.write_text('{}')
                with self.assertRaises(FileExistsError):
                    hcu_run._receipt_paths('results/job.json')

    def test_busy_hcu_never_starts_container(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            argv = ['hcu_run.py', '--image', 'pinned-image', '--device', '0',
                    '--receipt', 'results/job.json', '--', 'python3', 'work.py']

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
            self.assertFalse((root / 'results/job.json').exists())

    def test_zero_exit_with_busy_post_state_is_not_qualified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            argv = ['hcu_run.py', '--image', 'pinned-image', '--device', '0',
                    '--receipt', 'results/job.json', '--timeout', '20', '--', 'python3', 'work.py']
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
                    return subprocess.CompletedProcess(command, 0)
                raise AssertionError(command)

            with mock.patch.object(hcu_run, 'ROOT', root), \
                 mock.patch.object(hcu_run, 'LOCK_DIR', root / '.locks'), \
                 mock.patch.object(sys, 'argv', argv), \
                 mock.patch.object(hcu_run.subprocess, 'check_output', side_effect=output), \
                 mock.patch.object(hcu_run.subprocess, 'run', side_effect=run), \
                 mock.patch.object(hcu_run.time, 'sleep'):
                self.assertEqual(hcu_run.main(), 75)
            terminal = json.loads((root / 'results/job-terminal.json').read_text())
            self.assertEqual(terminal['status'], 'not_qualified')
            self.assertEqual(terminal['after_vram'], '97%')
            self.assertFalse(terminal['physical_exclusivity'])


if __name__ == '__main__':
    unittest.main()
