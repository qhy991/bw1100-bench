"""The profiling entry takes its image from the Run binding, never from the caller.

R6 retyped the digest with one wrong character and R9 pasted the usage placeholder
`IMAGE`; each reached `docker image inspect` and stopped the whole HCU.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from scripts.pinned_image import pinned_image

ROOT = Path(__file__).resolve().parents[1]
PINNED = 'sha256:' + '3ad0ae71' * 8


def run_root(temporary, binding='campaign/development-binding.json', image=PINNED):
    root = Path(temporary)
    (root / 'scripts').mkdir()
    for name in ('rocprof.sh', 'pinned_image.py', 'verify_rocprof_csv.py'):
        shutil.copy(ROOT / 'scripts' / name, root / 'scripts' / name)
    (root / 'scripts/__init__.py').write_text('')
    # A stub gateway records the image it was handed and stops: no GPU, no docker.
    (root / 'scripts/dtk.sh').write_text('printf "%s\\n" "$2" > "$(dirname "$0")/../dtk-image.txt"; exit 7\n')
    (root / binding).parent.mkdir(parents=True, exist_ok=True)
    (root / binding).write_text(json.dumps({'image': image, 'hcu': 5}))
    (root / 'pmc.txt').write_text('pmc: Wavefronts GPUBusy\n')
    return root


def rocprof(root, *arguments):
    environment = dict(os.environ, HIP_VISIBLE_DEVICES='5')
    return subprocess.run(['bash', str(root / 'scripts/rocprof.sh'), *arguments],
                          cwd=root, env=environment, capture_output=True, text=True)


class PinnedImageTests(unittest.TestCase):
    def test_reads_development_then_bench_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(pinned_image(run_root(temporary)), PINNED)
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(pinned_image(run_root(temporary, 'campaign/binding.json')), PINNED)

    def test_refuses_a_binding_without_an_exact_digest(self):
        for image in ('IMAGE', PINNED[:-1], PINNED.upper(), None):
            with self.subTest(image=image), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaisesRegex(ValueError, 'exact sha256'):
                    pinned_image(run_root(temporary, image=image))
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, 'no Run binding'):
                pinned_image(temporary)

    def test_gateway_receives_the_bound_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = run_root(temporary)
            result = rocprof(root, 'admission.json', 'pmc.txt', 'reports/metrics.csv',
                             'Kernel', 'python3', 'harness.py')
            self.assertEqual(result.returncode, 7, result.stderr)
            self.assertEqual((root / 'dtk-image.txt').read_text().strip(), PINNED)

    def test_a_caller_supplied_image_never_reaches_the_gateway(self):
        # The exact R9 call shape, and a retyped digest like R6's.
        for image in ('IMAGE', 'sha256:' + '3ad0ae71' * 7 + '3ad0ae76'):
            with self.subTest(image=image), tempfile.TemporaryDirectory() as temporary:
                root = run_root(temporary)
                result = rocprof(root, image, 'admission.json', 'pmc.txt',
                                 'reports/metrics.csv', 'Kernel', 'python3', 'harness.py')
                self.assertEqual(result.returncode, 2)
                self.assertIn('no longer takes an image', result.stderr)
                self.assertFalse((root / 'dtk-image.txt').exists())

    def test_missing_binding_stops_before_the_gateway(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = run_root(temporary)
            (root / 'campaign/development-binding.json').unlink()
            result = rocprof(root, 'admission.json', 'pmc.txt', 'reports/metrics.csv',
                             'Kernel', 'python3', 'harness.py')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('no Run binding', result.stderr)
            self.assertFalse((root / 'dtk-image.txt').exists())


if __name__ == '__main__':
    unittest.main()
