"""CPU rounding/ABI checks; the actual FlagGems device route needs qualification."""
import ast
import math
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


def activation_fp32(gate):
    value = gate.float()
    return 0.5 * value * (1 + torch.tanh(math.sqrt(2 / math.pi) *
                                       (value + 0.044715 * value.pow(3))))


def baseline_with_cpu_gelu(gelu):
    # Execute the actual adapter body without importing the GPU dependency.
    # Dependency identity/import and the original full gate are separate checks.
    path = ROOT / 'baselines/l1_048_flaggems_gelu.py'
    body = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == 'run')
    namespace = {'torch': torch, 'F': F, '_gelu': gelu}
    exec(compile(ast.Module(body=[body], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['run']


class GateUpBaselineTests(unittest.TestCase):
    def test_activation_round_precedes_multiply(self):
        x = torch.ones((1, 1, 1), dtype=torch.bfloat16)
        gate_weight = torch.tensor([[-3.0]], dtype=torch.bfloat16)
        up_weight = torch.tensor([[0.7]], dtype=torch.bfloat16)
        calls = []

        def gelu(gate, *, approximate):
            calls.append((gate.dtype, approximate))
            # Returning FP32 makes the adapter's explicit rounding observable.
            return activation_fp32(gate)

        observed = baseline_with_cpu_gelu(gelu)(x, gate_weight, up_weight)
        gate, up = F.linear(x, gate_weight), F.linear(x, up_weight)
        activated = activation_fp32(gate)
        expected = activated.to(torch.bfloat16) * up
        omitted_round = (activated * up.float()).to(torch.bfloat16)
        self.assertFalse(torch.equal(expected, omitted_round))
        self.assertTrue(torch.equal(observed, expected))
        self.assertEqual(calls, [(torch.bfloat16, 'tanh')])

    def test_batched_abi_and_input_ownership_with_bf16_community_output(self):
        x = (torch.arange(24).reshape(2, 3, 4) / 32 - 0.25).to(torch.bfloat16)
        gate_weight = (torch.arange(20).reshape(5, 4) / 32 - 0.25).to(torch.bfloat16)
        up_weight = gate_weight.flip(0).contiguous()
        before = [value.clone() for value in (x, gate_weight, up_weight)]
        run = baseline_with_cpu_gelu(lambda gate, approximate: activation_fp32(gate).to(gate.dtype))
        output = run(x, gate_weight, up_weight)
        expected = activation_fp32(F.linear(x, gate_weight)).to(torch.bfloat16) * F.linear(x, up_weight)
        self.assertEqual(output.shape, (2, 3, 5))
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(output, expected))
        self.assertTrue(all(torch.equal(value, saved) for value, saved in zip((x, gate_weight, up_weight), before)))
        saved_output = output.clone()
        x.zero_()
        changed = run(x, gate_weight, up_weight)
        self.assertTrue(torch.equal(output, saved_output))
        self.assertFalse(torch.equal(output, changed))


if __name__ == '__main__':
    unittest.main()
