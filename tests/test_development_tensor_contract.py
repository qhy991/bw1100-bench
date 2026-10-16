"""Task-owned tensor words and named geometry at the development boundary."""
import copy
import importlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SOURCE = os.environ.get('BW_TASK_SOURCE')
if SOURCE:
    sys.path.insert(0, str(Path(SOURCE) / 'src'))

from development.development_task_source import task_source

try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    from development.development_device import input_tensor, native_outputs


@unittest.skipUnless(SOURCE, 'needs the pinned task source')
class NamedGeometryTests(unittest.TestCase):
    def test_named_workloads_preserve_the_complete_original_documents(self):
        from open_cake_ir.tasks.workloads import WorkloadContract

        for task, count in [('aka_nearest_scatter', 23), ('aka_maxpool3d_backward', 21)]:
            owner = importlib.import_module('open_cake_ir.tasks.' + task)
            original = owner.workload_document(backend='triton-dcu')
            contract = dict(rows=None, columns=None, depth=None, workload=original)
            actual, source = task_source(task, contract)
            self.assertEqual(actual, original)
            self.assertEqual(source, owner.starter_source(WorkloadContract(original)))
            self.assertEqual(len(WorkloadContract(actual).case_ids), count)
            changed = copy.deepcopy(contract)
            changed['workload']['validation']['all_cases_required'] = False
            with self.assertRaises(ValueError):
                task_source(task, changed)

    def test_legacy_factory_still_rejects_changed_original_workload(self):
        from open_cake_ir.tasks.workloads import create_task

        doc, expected = create_task('rmsnorm', backend='triton-dcu', rows=2, columns=32)
        contract = dict(rows=2, columns=32, depth=None, workload=doc)
        self.assertEqual(task_source('rmsnorm', contract), (doc, expected))
        changed = copy.deepcopy(contract)
        changed['workload']['validation']['all_cases_required'] = False
        with self.assertRaises(ValueError):
            task_source('rmsnorm', changed)


@unittest.skipUnless(torch is not None, 'needs CPU PyTorch')
class TensorSnapshotTests(unittest.TestCase):
    def test_word_snapshots_preserve_signaling_nan_payload_and_zero_sign(self):
        words = [0x7FA12345, 0xFFC54321, 0x80000000, 0x00000001]
        signed = [x if x < 2**31 else x - 2**32 for x in words]
        output = torch.tensor(signed, dtype=torch.int32).view(torch.float32)
        workload = SimpleNamespace(document={'semantics': {'output_storage': 'fresh'}},
            tensor_abi=lambda case: [SimpleNamespace(name='y', dtype='fp32',
                                                     shape=(4,), mode='output')])
        for names in (None, ['y']):
            observed, metadata = native_outputs(workload, 'primary', output, device='cpu',
                                                names=names, expected_outputs={'y': output})
            self.assertEqual(observed['y'].view(torch.int32).tolist(), signed)
            self.assertNotEqual(observed['y'].data_ptr(), output.data_ptr())
            self.assertEqual(metadata['y']['shape'], [4])
            self.assertEqual(tuple(observed['y'].shape), (4,))
        legacy, _ = native_outputs(workload, 'primary', output, device='cpu')
        self.assertIsInstance(legacy['y'], list)

    def test_tensor_inputs_do_not_hide_wrong_rank_or_noncontiguity(self):
        arg = SimpleNamespace(name='x', dtype='fp32', shape=(4,))
        with self.assertRaises(ValueError):
            input_tensor(torch.ones(2, 2), arg)
        with self.assertRaises(ValueError):
            input_tensor(torch.ones(8)[::2], arg)
        self.assertEqual(tuple(input_tensor([1, 2, 3, 4], arg).shape), (4,))


if __name__ == '__main__':
    unittest.main()
