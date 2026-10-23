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

    def test_other_processes_kernels_are_not_counted(self):
        rows = self.launches([3_000] * 40)
        rows.insert(5, row('at::native::distribution_elementwise_grid_stride_kernel', 1, 2, 9_000_000))
        result = kernel_time.summarize(rows, {'_mma_r16_c64_k64_w4_s1_kernel'}, 30)
        self.assertEqual(result['median_us'], 3.0)

    def test_a_record_count_that_does_not_split_is_refused(self):
        with self.assertRaises(ValueError):
            kernel_time.summarize(self.launches([3_000] * 39), {'_mma_r16_c64_k64_w4_s1_kernel'}, 30)


class OwnerKernelTimePass(unittest.TestCase):
    """Run the owner with its subprocesses replaced; inspect what it asks rocprof to do."""

    def run_owner(self, profiled_code=0):
        import json, runpy, shutil, subprocess, sys, tempfile
        temp = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, temp)
        (temp / 'campaign/evaluations/e1').mkdir(parents=True)
        for name in ('development_evaluate_owner.py', 'development_stop.py'):
            shutil.copyfile(ROOT / 'development' / name, temp / 'campaign' / name)
        (temp / 'campaign/development-binding.json').write_text(json.dumps(dict(
            image='img', hcu=5, batch_stop_file=str(temp / 'STOP.json'))))
        (temp / 'campaign/deadline.json').write_text(json.dumps(dict(
            search_stop_at_epoch=10**12, stop_at_epoch=10**12)))
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
                return profiled_code
            return 0

        old = (subprocess.call, sys.argv[:], sys.path[:], Path.cwd())
        subprocess.call = call
        sys.argv = ['owner', '--candidate', 'c.py', '--id', 'e1']
        sys.path.insert(0, str(temp / 'campaign'))
        try:
            with self.assertRaises(SystemExit) as stop:
                runpy.run_path(str(temp / 'campaign/development_evaluate_owner.py'), run_name='__main__')
        finally:
            subprocess.call, sys.argv, sys.path = old[0], old[1], old[2]
            import os; os.chdir(old[3])
            sys.modules.pop('development_stop', None)
        return stop.exception.code, calls, json.loads((folder / 'outcome.json').read_text()), folder

    def test_profiled_pass_records_kernel_time_with_a_txt_pmc_input(self):
        code, calls, outcome, folder = self.run_owner()
        self.assertEqual(code, 0)
        [profiled] = [argv for argv in calls if any('rocprof' in part for part in argv)]
        pmc = profiled[profiled.index('-i') + 1]
        self.assertTrue(pmc.endswith('.txt'), pmc)
        self.assertEqual((folder / Path(pmc).name).read_text(), 'pmc: GPUBusy\n')
        self.assertIn('--timestamp', profiled)
        self.assertEqual(outcome['kernel_time'], {'median_us': 9.4})
        self.assertEqual(outcome['status'], 'accepted')

    def test_a_failed_profiled_pass_keeps_the_accepted_outcome(self):
        code, calls, outcome, _ = self.run_owner(profiled_code=1)
        self.assertEqual(code, 0)
        self.assertEqual(outcome['status'], 'accepted')
        self.assertIsNone(outcome['kernel_time'])
        self.assertIn('exited 1', outcome['kernel_time_unavailable'])
        self.assertFalse(any('--summarize' in argv for argv in calls))


if __name__ == '__main__':
    unittest.main()
