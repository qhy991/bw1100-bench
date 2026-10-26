import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from scripts.release_review import reviewed_logs,SCHEMA
from scripts.bench_queue import release_audit

class ReviewTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
  self.admission={'schema':SCHEMA,'job_id':'owned-job','hcu':4}
  self.terminal=dict(self.admission,completed_at=1,exit_code=1,container_still_running=False,after_vram='0%',after_kfd_visible=False)
  for name,data in [('receipt.json',self.admission),('receipt-terminal.json',self.terminal)]:
   (self.root/name).write_text(json.dumps(data))
  # Starts like a record, so only an explicit review can classify it (#42 exempts only text
  # that cannot be or hide a gateway record).
  (self.root/'stdout.json').write_text('{"warning":"runtime"}\n{"job_id":"owned-job","exit_code":1}\n{')
  self.review=dict(schema='bw1100-bench.diagnostic-log-review.v1',authorization='Explicit user authorization',runs=[dict(root=str(self.root),logs=[dict(path='stdout.json',sha256=self.digest('stdout.json'),receipt_pairs=[dict(admission='receipt.json',terminal='receipt-terminal.json',admission_sha256=self.digest('receipt.json'),terminal_sha256=self.digest('receipt-terminal.json'))])])])
 def digest(self,name):return hashlib.sha256((self.root/name).read_bytes()).hexdigest()
 def test_exact_log_preserves_failed_exit_and_confirms_release(self):
  self.assertFalse(release_audit(self.root)['release_records_clean'])
  exclusions=reviewed_logs(self.root,self.review)
  self.assertEqual(exclusions,['stdout.json'])
  self.assertTrue(release_audit(self.root,exclusions)['release_records_clean'])
  self.assertEqual(json.loads((self.root/'receipt-terminal.json').read_text())['exit_code'],1)
 def test_plain_captured_output_needs_no_review(self):
  (self.root/'stdout.json').write_text('runtime warning\n{"job_id":"owned-job","exit_code":1}\n')
  audit=release_audit(self.root)
  self.assertTrue(audit['release_records_clean'])
  self.assertEqual(audit['author_diagnostic_files'],['stdout.json'])
 def test_changed_file_hash_refused(self):
  (self.root/'stdout.json').write_text('changed')
  with self.assertRaises(ValueError):reviewed_logs(self.root,self.review)
 def test_admission_schema_log_never_excluded(self):
  (self.root/'stdout.json').write_text('{"schema":"'+SCHEMA+'","job_id":"owned-job",')
  self.review['runs'][0]['logs'][0]['sha256']=self.digest('stdout.json')
  with self.assertRaises(ValueError):reviewed_logs(self.root,self.review)
 def test_unknown_job_never_excluded(self):
  (self.root/'stdout.json').write_text('{"job_id":"foreign-job"}')
  self.review['runs'][0]['logs'][0]['sha256']=self.digest('stdout.json')
  with self.assertRaises(ValueError):reviewed_logs(self.root,self.review)
 def test_release_fields_and_identity_required(self):
  for change in ['after_vram','job_id','completed_at']:
   v=copy.deepcopy(self.terminal)
   if change=='completed_at':v.pop(change)
   else:v[change]='wrong'
   (self.root/'receipt-terminal.json').write_text(json.dumps(v))
   self.review['runs'][0]['logs'][0]['receipt_pairs'][0]['terminal_sha256']=self.digest('receipt-terminal.json')
   with self.subTest(change=change),self.assertRaises(ValueError):reviewed_logs(self.root,self.review)
 def test_unreviewed_partial_receipt_remains_blocked(self):
  (self.root/'another.json').write_text('{"schema":"'+SCHEMA+'",')
  self.assertFalse(release_audit(self.root,reviewed_logs(self.root,self.review))['release_records_clean'])
 def test_no_authorization_refused(self):
  self.review['authorization']=''
  with self.assertRaises(ValueError):reviewed_logs(self.root,self.review)
 def test_foreign_root_has_no_exclusions(self):
  self.review['runs'][0]['root']=str(self.root/'foreign')
  self.assertEqual(reviewed_logs(self.root,self.review),[])
