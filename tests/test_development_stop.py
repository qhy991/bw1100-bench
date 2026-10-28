"""Classify the observed DTK allocation refusal without hiding candidate failures."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from development.development_stop import (HCU_PCI_MISMATCH, HCU_PCI_MISMATCH_STOP,
                                          device_failure_is_infrastructure, halt)


class DevelopmentStopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.folder = self.root / 'evaluation'
        self.folder.mkdir()
        self.admission = {'schema': 'bw1100-bench.hcu-admission.v1', 'job_id': 'fixture'}
        (self.folder / 'admission.json').write_text(json.dumps(self.admission))
        self.after = dict(self.admission, completed_at=1, exit_code=125, container_still_running=False,
                          after_vram='0%', after_kfd_visible=False)
        (self.folder / 'admission-terminal.json').write_text(json.dumps(self.after))

    def tearDown(self):
        self.temp.cleanup()

    def test_dtk_unavailable_after_clean_release_requests_existing_stop(self):
        (self.folder / 'device.log').write_text('OCI runtime create failed: device with id=6 is unavailable\n')
        reason = device_failure_is_infrastructure(self.folder)
        self.assertIn('device runtime or admission service failed', reason)
        stop = self.root / 'STOP.json'
        halt({'batch_stop_file': str(stop), 'task': 'fixture'}, reason, 'evaluation')
        first = stop.read_text()
        halt({'batch_stop_file': str(stop), 'task': 'fixture'}, 'later failure', 'other')
        self.assertEqual(stop.read_text(), first)
        self.assertEqual(json.loads(first)['reason'], reason)

    def test_candidate_failures_generic_unavailability_and_exit_125_are_not_reclassified(self):
        for text in ('Triton compilation failed', 'Numerical comparison failed',
                     'subprocess returned exit status 125', 'device is unavailable',
                     'device with id=x is unavailable', 'device with id=6 might be unavailable'):
            with self.subTest(text=text):
                (self.folder / 'device.log').write_text(text)
                self.assertIsNone(device_failure_is_infrastructure(self.folder))

    def test_lock_wait_timeout_before_admission_does_not_stop_the_hcu(self):
        # round8-expanded59 lost HCU 5 to exactly this: a sibling author held the
        # per-HCU lock past the 300 s queue bound, nothing was admitted, and the
        # whole card was stopped. It is a resource wait for one evaluation.
        (self.folder / 'admission.json').unlink()
        (self.folder / 'admission-terminal.json').unlink()
        (self.folder / 'device.log').write_text(
            'Traceback (most recent call last):\n'
            "  File \"scripts/hcu_run.py\", line 103, in _acquire_hcu_lock\n"
            "    raise RuntimeError('timed out waiting for this HCU lock; no GPU admitted')\n"
            'RuntimeError: timed out waiting for this HCU lock; no GPU admitted\n')
        self.assertIsNone(device_failure_is_infrastructure(self.folder))

    def test_lock_text_after_an_admission_keeps_the_release_check(self):
        # Once admission.json exists the evaluation touched the device, so the
        # unknown-release rule still owns the decision.
        (self.folder / 'admission-terminal.json').unlink()
        (self.folder / 'device.log').write_text('timed out waiting for this HCU lock; no GPU admitted')
        self.assertEqual(device_failure_is_infrastructure(self.folder),
                         'device admission has no readable terminal release observation')

    def test_other_runtime_failures_before_admission_still_stop(self):
        (self.folder / 'admission.json').unlink()
        (self.folder / 'admission-terminal.json').unlink()
        (self.folder / 'device.log').write_text('Cannot connect to the Docker daemon at unix:///var/run/docker.sock')
        self.assertIn('device runtime or admission service failed',
                      device_failure_is_infrastructure(self.folder))

    def test_a_renumbered_card_stops_the_card_by_name_before_admission(self):
        (self.folder / 'admission.json').unlink()
        (self.folder / 'admission-terminal.json').unlink()
        (self.folder / 'device.log').write_text(
            '{"status": "refused_before_admission", "reason": "hcu_pci_mismatch"}\n'
            'RuntimeError: ' + HCU_PCI_MISMATCH + '\n')
        self.assertEqual(device_failure_is_infrastructure(self.folder), HCU_PCI_MISMATCH_STOP)

    def test_mismatch_text_that_is_not_the_last_gateway_line_is_not_a_renumbering(self):
        (self.folder / 'admission.json').unlink()
        (self.folder / 'admission-terminal.json').unlink()
        (self.folder / 'device.log').write_text(HCU_PCI_MISMATCH + '\nTriton compilation failed\n')
        self.assertNotEqual(device_failure_is_infrastructure(self.folder), HCU_PCI_MISMATCH_STOP)

    def test_a_mismatch_line_after_an_admission_is_not_a_renumbering(self):
        # Once a receipt exists the card was admitted, so the release check owns the decision.
        (self.folder / 'device.log').write_text('RuntimeError: ' + HCU_PCI_MISMATCH + '\n')
        self.assertNotEqual(device_failure_is_infrastructure(self.folder), HCU_PCI_MISMATCH_STOP)

    def test_unknown_release_takes_precedence_over_runtime_marker(self):
        (self.folder / 'device.log').write_text('device with id=6 is unavailable')
        after = self.folder / 'admission-terminal.json'
        after.unlink()
        self.assertEqual(device_failure_is_infrastructure(self.folder),
                         'device admission has no readable terminal release observation')
        self.after['container_still_running'] = True
        after.write_text(json.dumps(self.after))
        self.assertEqual(device_failure_is_infrastructure(self.folder), 'device release is unknown or incomplete')


if __name__ == '__main__':
    unittest.main()
