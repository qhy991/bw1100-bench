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
                if command == ['hy-smi', '--showpids']:
                    # Busy with no visible holder stays an unknown release.
                    return 'No KFD PIDs currently running!\n'
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
                 mock.patch.object(hcu_run, '_runtime_binding', return_value={'device_selector': '0', 'pci_bus_id': '0000:01:00.0'}), \
                 mock.patch.object(hcu_run.subprocess, 'check_output', side_effect=output), \
                 mock.patch.object(hcu_run.subprocess, 'run', side_effect=run), \
                 mock.patch.object(hcu_run.time, 'sleep'), \
                 mock.patch.object(hcu_run.time, 'monotonic', side_effect=(0, 31)):
                self.assertEqual(hcu_run.main(), 75)
            terminal = json.loads((root / 'results/job-terminal.json').read_text())
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

    def test_a_foreign_holder_after_exit_is_attributed_not_an_unknown_release(self):
        """r3 STOP-hcu3: another user's process took the card during the release window."""
        showpids = ("PIDs for KFD processes:\nPID: 1616893\n\tPASID: 32768\n\tHCU Index: ['3'] \n"
                    "\tVRAM USED(MiB): 5488\nPID: 1622073\n\tHCU Index: \n")
        outputs = iter(('', showpids) * 2)
        with mock.patch.object(hcu_run, '_vram', side_effect=('3%', '3%')), \
             mock.patch.object(hcu_run, '_kfd_visible', return_value=False), \
             mock.patch.object(hcu_run.subprocess, 'check_output', side_effect=lambda *a, **k: next(outputs)), \
             mock.patch.object(hcu_run.time, 'sleep'), \
             mock.patch.object(hcu_run.time, 'monotonic', side_effect=(0, 1, 31)):
            vram, kfd, live, observations = hcu_run._observe_release(3, 'owned-job', 0)
        self.assertEqual((vram, kfd, live), ('0%', False, False))
        self.assertEqual(observations[-1]['attributed_to_foreign_holders'], [1616893])

    def test_vram_with_no_visible_holder_is_still_an_unknown_release(self):
        with mock.patch.object(hcu_run, '_vram', side_effect=('3%', '3%')), \
             mock.patch.object(hcu_run, '_kfd_visible', return_value=False), \
             mock.patch.object(hcu_run.subprocess, 'check_output',
                               side_effect=('', 'No KFD PIDs currently running!\n', '', 'No KFD PIDs currently running!\n')), \
             mock.patch.object(hcu_run.time, 'sleep'), \
             mock.patch.object(hcu_run.time, 'monotonic', side_effect=(0, 1, 31)):
            vram, _, _, observations = hcu_run._observe_release(3, 'owned-job', 0)
        self.assertEqual(vram, '3%')
        self.assertNotIn('attributed_to_foreign_holders', observations[-1])

    def test_a_holder_on_another_card_is_not_attributed(self):
        showpids = "PID: 99\n\tHCU Index: ['5'] \n"
        with mock.patch.object(hcu_run, '_vram', side_effect=('3%', '3%')), \
             mock.patch.object(hcu_run, '_kfd_visible', return_value=False), \
             mock.patch.object(hcu_run.subprocess, 'check_output', side_effect=('', showpids, '', showpids)), \
             mock.patch.object(hcu_run.time, 'sleep'), \
             mock.patch.object(hcu_run.time, 'monotonic', side_effect=(0, 1, 31)):
            vram, _, _, _ = hcu_run._observe_release(3, 'owned-job', 0)
        self.assertEqual(vram, '3%')

    def test_a_device_visibility_override_is_refused_before_admission(self):
        """r3 STOP-hcu2: HIP_VISIBLE_DEVICES=2 inside the container hid the leased card."""
        for command in (['bash', '-c', 'HIP_VISIBLE_DEVICES=2 python3 run.py'],
                        ['env', 'ROCR_VISIBLE_DEVICES=0', 'python3', 'x.py'],
                        ['bash', '-c', 'export CUDA_VISIBLE_DEVICES=1; python3 x.py']):
            with self.subTest(command=command):
                self.assertIsNotNone(hcu_run._visibility_override(command))
        for command in (['python3', 'run.py'], ['bash', '-c', 'echo $HIP_VISIBLE_DEVICES'],
                        ['python3', '-c', 'import os; print(os.environ.get("MY_HIP_VISIBLE_DEVICES_X"))']):
            with self.subTest(command=command):
                self.assertIsNone(hcu_run._visibility_override(command))
        argv = ['hcu_run.py', '--image', 'img', '--device', '2', '--receipt', 'r.json', '--',
                'bash', '-c', 'HIP_VISIBLE_DEVICES=2 python3 run.py']
        with mock.patch.object(hcu_run.sys, 'argv', argv), \
             mock.patch.object(hcu_run, '_receipt_paths') as receipts, \
             mock.patch.object(hcu_run, '_stop_for_unknown_release') as stop:
            self.assertEqual(hcu_run.main(), 64)
        receipts.assert_not_called()
        stop.assert_not_called()

    def test_runtime_selector_joins_pci_instead_of_reusing_hcu_ordinal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'renderD131').mkdir()
            (root / 'renderD131/device').symlink_to(root / '0000:05:00.0')
            bus = 'HCU[3] : PCI Bus: 0000:05:00.0 --> SN: observed'
            inventory = {'kind': 'c-3000.com/hcu', 'devices': [
                {'name': '3', 'containerEdits': {'deviceNodes': [{'path': '/dev/dri/renderD131'}]}}]}
            binding = hcu_run._select_runtime_device(3, bus, inventory, root)
            self.assertEqual(binding['device_selector'], '3')
            self.assertEqual(binding['pci_bus_id'], '0000:05:00.0')
            with self.assertRaisesRegex(RuntimeError, 'does not uniquely match'):
                hcu_run._select_runtime_device(3, bus, {'kind': 'c-3000.com/hcu', 'devices': []}, root)

    def test_direct_profiler_runtime_error_is_distinct_from_candidate_error(self):
        self.assertEqual(hcu_run._runtime_error_marker('RuntimeError: No CUDA GPUs are available'),
                         'no cuda gpus are available')
        self.assertEqual(hcu_run._runtime_error_marker('RuntimeError: 0 active drivers ([])'),
                         '0 active drivers')
        self.assertEqual(hcu_run._runtime_error_marker('hsa_init failed: HSA_STATUS_ERROR_OUT_OF_RESOURCES'),
                         'hsa_status_error_out_of_resources')
        self.assertIsNone(hcu_run._runtime_error_marker('OutOfResources: shared memory required 98304'))
        self.assertIsNone(hcu_run._runtime_error_marker(
            'Ignoring [] DCUs as they are invalid\nOutOfResources: shared memory required 98304'))

    def test_binding_failure_stops_before_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stop = root / 'STOP.json'
            argv = ['hcu_run.py', '--image', 'image', '--device', '1',
                    '--receipt', 'results/job.json', '--', 'true']
            with mock.patch.object(hcu_run, 'ROOT', root), \
                 mock.patch.object(hcu_run, 'LOCK_DIR', root / '.locks'), \
                 mock.patch.dict(hcu_run.os.environ, {'BWBENCH_STOP_FILE': str(stop)}), \
                 mock.patch.object(sys, 'argv', argv), \
                 mock.patch.object(hcu_run.subprocess, 'check_output', return_value='sha256:pinned'), \
                 mock.patch.object(hcu_run, '_vram', return_value='0%'), \
                 mock.patch.object(hcu_run, '_kfd_visible', return_value=False), \
                 mock.patch.object(hcu_run, '_runtime_binding', side_effect=RuntimeError('missing PCI')):
                with self.assertRaisesRegex(RuntimeError, 'missing PCI'):
                    hcu_run.main()
            self.assertTrue(stop.exists())
            self.assertFalse((root / 'results/job.json').exists())

    def test_release_observer_denial_writes_unknown_terminal_and_stops_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stop = root / 'STOP.json'
            argv = ['hcu_run.py', '--image', 'image', '--device', '1',
                    '--receipt', 'results/job.json', '--', 'true']
            with mock.patch.object(hcu_run, 'ROOT', root), \
                 mock.patch.object(hcu_run, 'LOCK_DIR', root / '.locks'), \
                 mock.patch.dict(hcu_run.os.environ, {'BWBENCH_STOP_FILE': str(stop)}), \
                 mock.patch.object(sys, 'argv', argv), \
                 mock.patch.object(hcu_run.subprocess, 'check_output', return_value='sha256:pinned'), \
                 mock.patch.object(hcu_run.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as run, \
                 mock.patch.object(hcu_run, '_vram', return_value='0%'), \
                 mock.patch.object(hcu_run, '_kfd_visible', return_value=False), \
                 mock.patch.object(hcu_run, '_runtime_binding', return_value={'device_selector': '1', 'pci_bus_id': '0000:56:00.0'}), \
                 mock.patch.object(hcu_run, '_observe_release', side_effect=PermissionError('docker.sock')), \
                 mock.patch.object(hcu_run.time, 'sleep'):
                self.assertEqual(hcu_run.main(), 75)
                command = run.call_args.args[0]
                self.assertIn('DCU_VISIBLE_DEVICES=1', command)
                self.assertFalse(any(part.startswith('HIP_VISIBLE_DEVICES=') for part in command))
                self.assertNotIn('--privileged', command)
                self.assertNotIn('DCU_CREATOR', ' '.join(command))
                with self.assertRaisesRegex(RuntimeError, 'stopped new GPU'):
                    hcu_run.main()
            terminal = json.loads((root / 'results/job-terminal.json').read_text())
            self.assertEqual(terminal['status'], 'not_qualified')
            self.assertIsNone(terminal['after_kfd_visible'])
            self.assertIsNone(terminal['container_still_running'])
            self.assertTrue(stop.exists())


if __name__ == '__main__':
    unittest.main()
