"""Device kernel time comes from the candidate's own kernels, not the callable floor."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('kernel_time', ROOT / 'development/development_kernel_time.py')
kernel_time = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(kernel_time)

SOURCE = '''
import triton
import triton.language as tl

@triton.jit
def _mma_r16_c64_k64_w4_s1_kernel(a, b):
    pass

@triton.jit(do_not_specialize=["n"])
def _cake_epilogue_kernel(c, n):
    pass

def mma_r16(a, b):
    _mma_r16_c64_k64_w4_s1_kernel[(1,)](a, b)
'''


def row(name, dispatch, begin, end):
    return {'KernelName': '"%s.kd"' % name, 'DispatchNs': str(dispatch),
            'BeginNs': str(begin), 'EndNs': str(end)}


class KernelNames(unittest.TestCase):
    def test_only_jit_functions_are_candidate_kernels(self):
        self.assertEqual(kernel_time.kernel_names(SOURCE),
                         {'_mma_r16_c64_k64_w4_s1_kernel', '_cake_epilogue_kernel'})


class Summary(unittest.TestCase):
    def launches(self, durations_ns, kernels=('_mma_r16_c64_k64_w4_s1_kernel',)):
        rows, t = [], 0
        for duration in durations_ns:
            for name in kernels:
                rows.append(row(name, t, t + 100, t + 100 + duration))
                t += 10_000
        return rows

    def test_warmup_is_dropped_and_median_is_the_timed_launches(self):
        warmup = [900_000] * kernel_time.WARMUP
        timed = [4_000 + 10 * i for i in range(30)]
        result = kernel_time.summarize(self.launches(warmup + timed),
                                       {'_mma_r16_c64_k64_w4_s1_kernel'}, 30)
        self.assertEqual(result['launches'], 30)
        self.assertEqual(result['min_us'], 4.0)
        self.assertEqual(result['median_us'], 4.15)

    def test_a_multi_kernel_launch_is_summed(self):
        names = ('_mma_r16_c64_k64_w4_s1_kernel', '_cake_epilogue_kernel')
        result = kernel_time.summarize(self.launches([2_000] * 40, names), set(names), 30)
        self.assertEqual(result['kernels_per_launch'], 2)
        self.assertEqual(result['median_us'], 4.0)

    def test_same_process_framework_kernels_are_not_counted(self):
        rows = self.launches([3_000] * 40)
        rows.insert(5, row('at::native::distribution_elementwise_grid_stride_kernel', 1, 2, 9_000_000))
        result = kernel_time.summarize(rows, {'_mma_r16_c64_k64_w4_s1_kernel'}, 30)
        self.assertEqual(result['median_us'], 3.0)

    def test_a_record_count_that_does_not_split_is_refused(self):
        with self.assertRaises(ValueError):
            kernel_time.summarize(self.launches([3_000] * 39), {'_mma_r16_c64_k64_w4_s1_kernel'}, 30)

    def test_uneven_per_kernel_counts_are_refused_even_when_the_total_splits(self):
        names = ('_mma_r16_c64_k64_w4_s1_kernel', '_cake_epilogue_kernel')
        rows = self.launches([2_000] * 40, names)
        rows.remove(next(r for r in rows if r['KernelName'].startswith('"_cake')))
        rows.append(row(names[0], 10**9, 10**9, 10**9 + 2_000))
        with self.assertRaises(ValueError):
            kernel_time.summarize(rows, set(names), 30)

    def test_a_launch_with_different_kernels_is_refused_when_counts_still_split(self):
        names = ('_mma_r16_c64_k64_w4_s1_kernel', '_cake_epilogue_kernel')
        rows = self.launches([2_000] * 40, names)
        rows[0]['KernelName'], rows[3]['KernelName'] = rows[1]['KernelName'], rows[0]['KernelName']
        with self.assertRaisesRegex(ValueError, 'same candidate kernels'):
            kernel_time.summarize(rows, set(names), 30)

    def test_a_refused_summary_leaves_no_record_under_campaign(self):
        """An empty kernel-time.json would read as an unreadable release receipt."""
        import csv, json, shutil, subprocess, sys, tempfile
        temp = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, temp)
        folder = temp / 'campaign/evaluations/e1'; folder.mkdir(parents=True)
        shutil.copyfile(ROOT / 'development/development_kernel_time.py', temp / 'campaign/development_kernel_time.py')
        (folder / 'generated.py').write_text(SOURCE)
        (folder / 'emission.json').write_text(json.dumps(dict(source='campaign/evaluations/e1/generated.py')))
        with (folder / 'kernel-time.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, ['KernelName', 'DispatchNs', 'BeginNs', 'EndNs'])
            writer.writeheader(); writer.writerow(row('other', 1, 2, 3))
        result = subprocess.run([sys.executable, 'campaign/development_kernel_time.py', '--emission',
                                 'campaign/evaluations/e1/emission.json', '--summarize',
                                 'campaign/evaluations/e1/kernel-time.csv', '--output',
                                 'campaign/evaluations/e1/kernel-time.json'], cwd=temp, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('_mma_r16', result.stderr)
        self.assertFalse((folder / 'kernel-time.json').exists())

    def test_no_candidate_kernel_names_the_expected_kernels(self):
        with self.assertRaisesRegex(ValueError, '_mma_r16'):
            kernel_time.summarize([row('other', 1, 2, 3)], {'_mma_r16_c64_k64_w4_s1_kernel'}, 30)


class OwnerKernelTimePass(unittest.TestCase):
    """Run the owner with its subprocesses replaced; inspect what it asks rocprof to do."""

    def run_owner(self, profiled_code=0, stop_at=10**12, profiled_effect=None, phase='search'):
        import json, runpy, shutil, subprocess, sys, tempfile
        temp = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, temp)
        (temp / 'campaign/evaluations/e1').mkdir(parents=True)
        for name in ('development_evaluate_owner.py', 'development_stop.py'):
            shutil.copyfile(ROOT / 'development' / name, temp / 'campaign' / name)
        (temp / 'campaign/development-binding.json').write_text(json.dumps(dict(
            image='img', hcu=5, batch_stop_file=str(temp / 'STOP.json'))))
        (temp / 'campaign/deadline.json').write_text(json.dumps(dict(
            search_stop_at_epoch=stop_at, stop_at_epoch=stop_at)))
        folder = temp / 'campaign/evaluations/e1'
        calls = []

        def call(argv, **kwargs):
            calls.append(argv)
            if '--phase' in argv and argv[argv.index('--phase') + 1] == 'emit':
                (folder / 'emission.json').write_text('{}')
            elif '--phase' in argv:
                (folder / 'outcome.json').write_text(json.dumps(dict(id='e1', status='accepted')))
            elif '--summarize' in argv:
                (folder / 'kernel-time.json').write_text(json.dumps(dict(median_us=9.4)))
            elif any('rocprof' in part for part in argv):
                if profiled_effect:
                    profiled_effect(folder)
                return profiled_code
            return 0

        old = (subprocess.call, sys.argv[:], sys.path[:], Path.cwd())
        subprocess.call = call
        sys.argv = ['owner', '--candidate', 'c.py', '--id', 'e1', '--phase', phase]
        sys.path.insert(0, str(temp / 'campaign'))
        try:
            with self.assertRaises(SystemExit) as stop:
                runpy.run_path(str(temp / 'campaign/development_evaluate_owner.py'), run_name='__main__')
        finally:
            subprocess.call, sys.argv, sys.path = old[0], old[1], old[2]
            import os; os.chdir(old[3])
            sys.modules.pop('development_stop', None)
        self.stop_file = temp / 'STOP.json'
        self.assertEqual(json.loads((folder / 'outcome.json').read_text()), dict(id='e1', status='accepted'))
        return stop.exception.code, calls, json.loads((folder / 'kernel-time-outcome.json').read_text()), folder

    def test_profiled_pass_records_kernel_time_with_a_txt_pmc_input(self):
        code, calls, outcome, folder = self.run_owner()
        self.assertEqual(code, 0)
        [profiled] = [argv for argv in calls if any('rocprof' in part for part in argv)]
        pmc = profiled[profiled.index('-i') + 1]
        self.assertTrue(pmc.endswith('.txt'), pmc)
        self.assertEqual((folder / Path(pmc).name).read_text(), 'pmc: GPUBusy\n')
        self.assertIn('--timestamp', profiled)
        self.assertEqual(outcome['kernel_time'], {'median_us': 9.4})
        receipt = profiled[profiled.index('--receipt') + 1]
        self.assertTrue(receipt.endswith('kernel-time-admission.json'), receipt)

    def test_a_failed_profiled_pass_keeps_the_accepted_outcome(self):
        code, calls, outcome, _ = self.run_owner(profiled_code=1)
        self.assertEqual(code, 0)
        self.assertIsNone(outcome['kernel_time'])
        self.assertIn('exited 1', outcome['kernel_time_unavailable'])
        self.assertFalse(any('--summarize' in argv for argv in calls))
        self.assertFalse(self.stop_file.exists())

    def test_an_unconfirmed_release_of_the_profiled_admission_stops_the_card(self):
        import json
        def admitted_without_terminal(folder):
            (folder / 'kernel-time-admission.json').write_text(json.dumps(dict(schema='s', job_id='j')))
        code, _, outcome, _ = self.run_owner(profiled_code=1, profiled_effect=admitted_without_terminal)
        self.assertEqual(code, 0)
        self.assertIn('no readable terminal', outcome['kernel_time_unavailable'])
        self.assertIn('no readable terminal', json.loads(self.stop_file.read_text())['reason'])

    def test_an_infrastructure_signature_in_the_profiled_log_stops_the_card(self):
        def unavailable(folder):
            with (folder / 'kernel-time.log').open('a') as log:
                log.write('OCI runtime create failed: device with id=5 is unavailable\n')
        code, _, outcome, _ = self.run_owner(profiled_code=1, profiled_effect=unavailable)
        self.assertIn('device runtime', outcome['kernel_time_unavailable'])
        self.assertTrue(self.stop_file.exists())

    def test_confirmation_waits_for_the_card_lock_while_its_phase_admits_it(self):
        import time
        stop_at = int(time.time()) + 1800
        for phase, longer in (('search', False), ('confirmation', True)):
            with self.subTest(phase=phase):
                _, calls, _, _ = self.run_owner(stop_at=stop_at, phase=phase)
                [device] = [argv for argv in calls if 'scripts/hcu_run.py' in argv and 'campaign/development_device.py' in argv]
                wait = int(device[device.index('--queue-timeout') + 1])
                latest = float(device[device.index('--latest-start-epoch') + 1])
                timeout = int(device[device.index('--timeout') + 1])
                self.assertEqual(wait > 300, longer)
                self.assertLessEqual(time.time() + wait, latest + 5)
                # The device job ends with room for its terminal and compare before the owner stop at stop-20 s.
                self.assertLessEqual(latest + timeout, stop_at - (150 if longer else 30))

    def test_too_little_time_skips_the_pass_with_its_reason(self):
        import time
        code, calls, outcome, _ = self.run_owner(stop_at=int(time.time()) + 100)
        self.assertEqual(code, 0)
        self.assertFalse(any(any('rocprof' in part for part in argv) for argv in calls))
        self.assertIn('before the phase stop', outcome['kernel_time_unavailable'])


if __name__ == '__main__':
    unittest.main()
