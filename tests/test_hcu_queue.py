"""Actual process locks verify serialization; no GPU or Docker is used here."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

MODULE=Path(__file__).resolve().parents[1]/'scripts/hcu_run.py'
spec=importlib.util.spec_from_file_location('queue_hcu',str(MODULE))
hcu=importlib.util.module_from_spec(spec);spec.loader.exec_module(hcu)

WORKER='''import importlib.util,json,os,select,sys,time
s=importlib.util.spec_from_file_location('hcu',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
fd=os.open(sys.argv[2],os.O_CREAT|os.O_RDWR,0o600)
try:
 wait=m._acquire_hcu_lock(fd,float(sys.argv[4]))
 started=time.monotonic()
 with open(sys.argv[3]+'.acquired','w') as f:json.dump({'started':started,'wait':wait},f)
 if sys.argv[5]=='release':
  if not select.select([sys.stdin],[],[],15)[0]:raise RuntimeError('holder release handshake timed out')
  if sys.stdin.readline()!='release\\n':raise RuntimeError('holder release handshake was not received')
 else:time.sleep(float(sys.argv[5]))
 finished=time.monotonic()
 with open(sys.argv[3]+'.finished','w') as f:json.dump({'started':started,'finished':finished,'wait':wait},f)
finally:os.close(fd)
'''

class HcuQueueTests(unittest.TestCase):
    def worker(self,lock,result,queue,hold):
        return subprocess.Popen([sys.executable,'-c',WORKER,str(MODULE),str(lock),str(result),str(queue),str(hold)],stdin=subprocess.PIPE if hold=='release' else None,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)

    def wait_file(self,path,worker):
        end=time.monotonic()+5
        while not path.exists():
            if worker.poll() is not None:self.fail(worker.communicate()[1])
            if time.monotonic()>end:self.fail('worker did not reach the observed ownership point')
            time.sleep(.01)

    def test_two_task_processes_share_one_card_lock_without_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);lock=root/'hcu-2.lock';a=root/'a';b=root/'b'
            first=self.worker(lock,a,2,.35)
            try:
                self.wait_file(Path(str(a)+'.acquired'),first)
                second=self.worker(lock,b,2,.02)
                try:
                    for worker in [first,second]:
                        _,error=worker.communicate(timeout=5);self.assertEqual(worker.returncode,0,error)
                    left=json.loads(Path(str(a)+'.finished').read_text());right=json.loads(Path(str(b)+'.finished').read_text())
                    self.assertGreaterEqual(right['started'],left['finished'])
                    self.assertGreater(right['wait'],0)
                finally:
                    if second.poll() is None:second.kill();second.communicate()
            finally:
                if first.poll() is None:first.kill();first.communicate()

    def test_timeout_and_legacy_failfast_cannot_enter_occupied_slot(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);lock=root/'hcu-1.lock';a=root/'holder'
            holder=self.worker(lock,a,2,'release')
            try:
                self.wait_file(Path(str(a)+'.acquired'),holder)
                for wait,reason in [(0,'another local'),(.1,'timed out waiting')]:
                    result=root/('wait'+str(wait));worker=self.worker(lock,result,wait,0)
                    _,error=worker.communicate(timeout=5)
                    self.assertNotEqual(worker.returncode,0);self.assertIn(reason,error)
                    self.assertFalse(Path(str(result)+'.acquired').exists())
            finally:
                try:
                    _,error=holder.communicate(input='release\n',timeout=5)
                    self.assertEqual(holder.returncode,0,error)
                finally:
                    if holder.poll() is None:holder.kill();holder.communicate()

    def test_expired_window_never_probes_or_launches_device(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            argv=['hcu_run.py','--image','image','--device','1','--receipt','results/job.json',
                  '--queue-timeout','1','--latest-start-epoch','1','--','python3','work.py']
            with mock.patch.object(hcu,'ROOT',root),mock.patch.object(hcu,'LOCK_DIR',root/'.locks'),mock.patch.object(sys,'argv',argv),mock.patch.object(hcu.subprocess,'check_output') as output,mock.patch.object(hcu.subprocess,'run') as run:
                with self.assertRaisesRegex(RuntimeError,'window expired'):hcu.main()
                output.assert_not_called();run.assert_not_called()
            self.assertFalse((root/'results/job.json').exists())

    def test_probe_time_cannot_move_admission_past_its_deadline(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);argv=['hcu_run.py','--image','image','--device','1','--receipt','results/job.json','--latest-start-epoch','10','--','python3','work.py']
            with mock.patch.object(hcu,'ROOT',root),mock.patch.object(hcu,'LOCK_DIR',root/'.locks'),mock.patch.object(sys,'argv',argv),mock.patch.object(hcu.time,'time',side_effect=[9,11]),mock.patch.object(hcu.subprocess,'check_output',return_value='sha256:image'),mock.patch.object(hcu,'_vram',return_value='0%'),mock.patch.object(hcu,'_kfd_visible',return_value=False),mock.patch.object(hcu.subprocess,'run') as run:
                with self.assertRaisesRegex(RuntimeError,'window expired'):hcu.main()
                run.assert_not_called()
            self.assertFalse((root/'results/job.json').exists())

    def test_nonfinite_or_unbounded_controls_are_rejected_before_device(self):
        for controls in [('--queue-timeout','-1'),('--queue-timeout','3601'),('--latest-start-epoch','nan'),('--latest-start-epoch','inf')]:
            argv=['hcu_run.py','--image','image','--device','1','--receipt','job.json']+list(controls)+['--','python3','work.py']
            with mock.patch.object(sys,'argv',argv),mock.patch.object(hcu.subprocess,'run') as run:
                with self.assertRaises(SystemExit):hcu.main()
                run.assert_not_called()

if __name__=='__main__':unittest.main()
