"""Focused acceptance checks for the diagnostic profile gate."""

import csv
from pathlib import Path
import tempfile
import unittest

from scripts.verify_rocprof_csv import requested_metrics, verify


class RocprofCsvTests(unittest.TestCase):
    def test_requires_requested_counters_and_target_kernel(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pmc = root / 'pmc.txt'
            report = root / 'metrics.csv'
            pmc.write_text('pmc: Wavefronts GPUBusy\n')
            with report.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['KernelName', 'Wavefronts', 'GPUBusy'])
                writer.writeheader()
                writer.writerow({'KernelName': 'other_kernel', 'Wavefronts': 1, 'GPUBusy': 20})
                writer.writerow({'KernelName': 'Rmsnorm2dFwd [clone .kd]',
                                 'Wavefronts': 524, 'GPUBusy': 100})
            result = verify(pmc, report, 'Rmsnorm2dFwd')
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(result['total_rows'], 2)
            self.assertEqual(result['matched_rows'], 1)
            with self.assertRaisesRegex(ValueError, 'no matching kernel'):
                verify(pmc, report, 'nonexistent')
            pmc.write_text('pmc: Wavefronts FETCH_SIZE\n')
            with self.assertRaisesRegex(ValueError, 'missing columns'):
                verify(pmc, report, 'Rmsnorm2dFwd')

    def test_rejects_counter_sets_that_exceed_one_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'pmc.txt'
            path.write_text('pmc: a b c d e f g\n')
            with self.assertRaisesRegex(ValueError, '1..6'):
                requested_metrics(path)


if __name__ == '__main__':
    unittest.main()
