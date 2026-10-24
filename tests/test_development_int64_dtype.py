"""INT64 ABI arguments at the development boundary.

`DTYPES` in `development/development_device.py` is the only dtype-string-to-torch
table in the harness, and it is consulted for every input argument (conversion and
validation) and for every output contract. It omitted `int64`, so a Task whose ABI
carries an INT64 input could not be prepared at all: `development_prepare.py` died
with `KeyError: 'int64'` at `development_device.py:24` before a single schedule was
assessed. `aka_signed_remainder` is such a Task -- its single-element INT64 divisor
is the point of the Task -- and it took a whole round down with it.

The Compiler requires that argument to arrive as a **native CPU tensor**
(`evaluation/core.py:873`, `LoadedTorchTensorCandidate`) precisely so the full
64-bit word survives: a Python float cannot carry `int64_min` or `int64_max`. These
tests pin that the adapter produces exactly that, and that the new map entry does
not weaken the validation that reads the same map.
"""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    from development.development_device import (DTYPES, input_tensor,
                                                native_output_contract)


def arg(name, shape, dtype, mode='input'):
    return SimpleNamespace(name=name, shape=shape, dtype=dtype, mode=mode)


@unittest.skipUnless(torch is not None, 'needs torch')
class Int64ArgumentTests(unittest.TestCase):
    def test_each_admitted_dtype_name_maps_to_one_distinct_torch_dtype(self):
        self.assertEqual(set(DTYPES), {'fp32', 'bf16', 'fp16', 'int32', 'int64'})
        self.assertEqual(len(set(DTYPES.values())), len(DTYPES))

    def test_int64_values_materialize_as_a_contiguous_cpu_int64_tensor(self):
        tensor = input_tensor([7, -7, 0], arg('d', (3,), 'int64'))
        self.assertEqual(tensor.dtype, torch.int64)
        self.assertEqual(tensor.device.type, 'cpu')
        self.assertTrue(tensor.is_contiguous())
        self.assertEqual(tensor.reshape(-1).tolist(), [7, -7, 0])

    def test_int64_extremes_survive_exactly(self):
        """The values a Python float cannot hold are the ones that must not round."""
        extreme = [-2 ** 63, 2 ** 63 - 1]
        tensor = input_tensor(extreme, arg('d', (2,), 'int64'))
        self.assertEqual(tensor.reshape(-1).tolist(), extreme)

    def test_a_materialized_int64_tensor_is_copied_not_rebuilt(self):
        source = torch.tensor([-2 ** 63], dtype=torch.int64)
        tensor = input_tensor(source, arg('d', (1,), 'int64'))
        self.assertEqual(tensor.dtype, torch.int64)
        self.assertEqual(tensor.reshape(-1).tolist(), [-2 ** 63])
        self.assertIsNot(tensor, source)

    def test_a_wrong_dtype_for_an_int64_abi_argument_is_still_refused(self):
        """The added entry must not turn the validation that reads it off."""
        with self.assertRaises(ValueError) as raised:
            input_tensor(torch.tensor([1.0], dtype=torch.float32), arg('d', (1,), 'int64'))
        self.assertIn('dtype differs', str(raised.exception))

    def test_an_int64_output_contract_names_torch_int64(self):
        workload = SimpleNamespace(tensor_abi=lambda case_id: [arg('out', (2,), 'int64', 'output')])
        self.assertEqual(native_output_contract(workload, 'primary')['out']['dtype'], 'torch.int64')


if __name__ == '__main__':
    unittest.main()
