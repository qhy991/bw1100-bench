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


if __name__ == '__main__':
    unittest.main()
