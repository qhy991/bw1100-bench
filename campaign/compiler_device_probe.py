"""Bounded generated-kernel device probe, not original-task qualification."""
import importlib.util
import json
from pathlib import Path
import torch

root = Path(__file__).resolve().parents[1]
path = root / 'campaign/results/compiler-preflight-rmsnorm.py'
spec = importlib.util.spec_from_file_location('cake_device_probe', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.manual_seed(200)
x = torch.randn((8, 512, 128), device='cuda', dtype=torch.float32)
gamma = torch.randn((128,), device='cuda', dtype=torch.float32)
expected = x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + 1e-6) * gamma
actual = module.cake_gfx938_rmsnorm_b8_smoke(x, gamma)
torch.cuda.synchronize()
torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
arch = torch.cuda.get_device_properties(0).gcnArchName.split(':', 1)[0]
if arch != 'gfx938':
    raise ValueError('unexpected target ' + arch)
with (root / 'campaign/results/compiler-device-probe.json').open('x') as stream:
    json.dump({'status': 'passed', 'arch': arch, 'values': x.numel(),
               'generated_kernel': str(path.relative_to(root)),
               'original_task_full_correctness': False, 'performance': 'not_measured'}, stream, indent=2)
print('Frozen Compiler emitted RMSNorm executed correctly on gfx938; no task-performance claim')
