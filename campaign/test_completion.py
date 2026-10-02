import hashlib,json,tempfile,unittest
from pathlib import Path
from campaign.completion import validate

class CompletionEvidence(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        for name in ('candidate.py','baseline.py'): (self.root/name).write_text('def run(): pass\n')
        sha=hashlib.sha256((self.root/'candidate.py').read_bytes()).hexdigest()
        self.plan={'tasks':[{'id':'task','baseline':'baseline.py','baseline_sha256':sha}]}
        gate={'full_device_correctness':True,'status':'passed','arch':'gfx938','task':'task','cases':[{'workload_uuid':str(u),'round':r,'passed':True} for u in range(16) for r in range(10)]}
        self.timing={'candidate_source_sha256':sha,'baseline_source_sha256':sha,'rows':[{'workload_uuid':str(u),'inputs_unmutated_after_measurement':True,'baseline_us':{'median':100,'forward_median':100,'reverse_median':100},'candidate_us':{'median':80,'forward_median':80,'reverse_median':80},'aa_control_baseline_us':{'abs_drift':1}} for u in range(16)]}
        (self.root/'gate.json').write_text(json.dumps(gate)); self.save()
        self.done={'tasks':[{'task':'task','status':'accepted','reason':'paired gain','evidence':['gate.json','timing.json'],'profile_skip_reason':'CPU dispatch hypothesis; see evidence','candidate_source':'candidate.py','correctness_source_sha256':sha,'correctness_report':'gate.json','latency_report':'timing.json'}]}
    def save(self): (self.root/'timing.json').write_text(json.dumps(self.timing))
    def test_complete(self): validate(self.root,self.done,self.plan)
    def test_stale_source(self):
        (self.root/'candidate.py').write_text('changed\n')
        with self.assertRaises(ValueError): validate(self.root,self.done,self.plan)
    def test_reverse_regression(self):
        self.timing['rows'][0]['candidate_us']['reverse_median']=120; self.save()
        with self.assertRaises(ValueError): validate(self.root,self.done,self.plan)
    def test_missing_task(self):
        with self.assertRaises(ValueError): validate(self.root,{'tasks':[]},self.plan)
    def test_escape(self):
        self.done['tasks'][0]['evidence']=['../outside']
        with self.assertRaises(ValueError): validate(self.root,self.done,self.plan)
    def test_negative_evidence(self):
        self.done['tasks'][0]['status']='no_robust_gain'; validate(self.root,self.done,self.plan)
if __name__=='__main__': unittest.main()
