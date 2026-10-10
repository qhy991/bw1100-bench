"""Real CPU process checks; the sibling CLI is a local fixture, never a Provider."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class AuthorEntryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.bin = self.root / 'environment/bin'
        self.bin.mkdir(parents=True)
        self.python = self.bin / 'python3'
        self.python.symlink_to(Path(sys.executable))
        self.entry = self.bin / 'hmz'
        self.run = self.root / 'run'
        self.run.mkdir()
        (self.run / 'campaign').mkdir()
        self.env = dict(os.environ, PATH=os.pathsep.join(('/usr/bin', '/bin')),
                        PYTHONDONTWRITEBYTECODE='1')

    def tearDown(self):
        self.temporary.cleanup()

    def process(self, code):
        result = subprocess.run([str(self.python), '-c', code, str(ROOT), str(self.run), str(self.root)],
                                env=self.env, check=True, text=True, capture_output=True, timeout=10)
        return json.loads(result.stdout.splitlines()[-1])

    def test_sibling_entry_works_without_path_entry_and_keeps_environment_bin(self):
        self.entry.write_text('#!/usr/bin/env python3\nimport json,os,sys\n'
                              'print(json.dumps({"argv":sys.argv,"python":sys.executable,"first_path":os.environ["PATH"].split(os.pathsep)[0]}))\n')
        self.entry.chmod(0o755)
        result = self.process('''
import json,os,shutil,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from campaign import launch
launch.ROOT=Path(sys.argv[2])
binding={'home':sys.argv[3],'hcu':1}
launch.reconcile=lambda root:binding
missing_on_path=shutil.which('hmz') is None
sys.argv=['launch','--check']
launch.main()
entry,env=launch.author_entry(binding)
with (launch.ROOT/'author.log').open('w') as stream:
    code=launch.author([str(entry),'fixture-exec'],env,5,stream)
observed=json.loads((launch.ROOT/'author.log').read_text())
print(json.dumps({'missing_on_path':missing_on_path,'entry':str(entry),'exit_code':code,'observed':observed}))
''')
        self.assertTrue(result['missing_on_path'])
        self.assertEqual(result['entry'], str(self.entry))
        self.assertEqual(result['exit_code'], 0)
        self.assertEqual(result['observed']['python'], str(self.python))
        self.assertEqual(result['observed']['first_path'], str(self.bin))
        self.assertEqual(result['observed']['argv'][1:], ['fixture-exec'])
        self.assertFalse((self.run / 'campaign/deadline.json').exists())
        self.assertFalse((self.run / 'campaign/intake.json').exists())

    def test_missing_nonexecutable_or_unusable_entry_fails_before_budget(self):
        for failure in ('missing', 'nonexecutable', 'unusable'):
            with self.subTest(failure=failure):
                if self.entry.exists(): self.entry.unlink()
                if failure != 'missing':
                    self.entry.write_text('#!/bin/sh\nexit 7\n')
                    self.entry.chmod(0o755 if failure == 'unusable' else 0o600)
                result = self.process('''
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from campaign import launch
launch.ROOT=Path(sys.argv[2])
binding={'home':sys.argv[3],'hcu':1}
launch.reconcile=lambda root:binding
sys.argv=['launch']
try:
    launch.main()
except (RuntimeError,OSError,launch.subprocess.SubprocessError) as error:
    print(json.dumps({'error':type(error).__name__}))
''')
                self.assertIn(result['error'], ('RuntimeError', 'CalledProcessError'))
                self.assertFalse((self.run / 'campaign/deadline.json').exists())
                self.assertFalse((self.run / 'campaign/intake.json').exists())


if __name__ == '__main__':
    unittest.main()
