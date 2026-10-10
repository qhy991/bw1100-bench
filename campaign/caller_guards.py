"""Original-oracle caller checks; separate from original16x10 numeric gate."""
from pathlib import Path
import argparse,json,hashlib,sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import bwbench
from sol_execbench.core.bench.correctness import set_seed
from sol_execbench.core.bench.io import gen_inputs

def snapshot(value):
 if isinstance(value,torch.Tensor):return value.detach().clone()
 if isinstance(value,dict):return {k:snapshot(v) for k,v in value.items()}
 if isinstance(value,tuple):return tuple(snapshot(v) for v in value)
 if isinstance(value,list):return [snapshot(v) for v in value]
 return value

def equal(a,b):
 if isinstance(a,torch.Tensor):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b

def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 intake=json.loads((ROOT/'campaign/intake.json').read_text());plan=json.loads((ROOT/intake['plan']).read_text());task=plan['tasks'][0]
 bwbench.upstream();d,ws,_,folder=bwbench.load_problem(bwbench.task_record(task['id']))
 w=next(x for x in ws if x.uuid==task['smoke_uuid']);ref=bwbench.load_module(folder/'reference.py','guard_reference');custom=getattr(ref,d.custom_inputs_entrypoint) if d.custom_inputs_entrypoint else None
 fn=bwbench.load_module(a.candidate,'guard_candidate').run;torch.set_num_threads(4)
 checks=[]
 report={'status':'running','task':task['id'],'workload_uuid':w.uuid,'candidate_source_sha256':hashlib.sha256(a.candidate.read_bytes()).hexdigest(),'scope':'one predeclared original workload; caller checks do not replace original16x10','checks':checks}
 with a.output.open('x') as f:json.dump(report,f)
 def save():
  temporary=a.output.with_suffix(a.output.suffix+'.tmp');temporary.write_text(json.dumps(report,indent=2));temporary.replace(a.output)
 def compare(name,inputs):
  cloned=bwbench.cloned_inputs(inputs);expected=ref.run(*cloned);del cloned
  before=snapshot(inputs);actual=fn(*inputs);torch.cuda.synchronize()
  result=bwbench.compare_outputs(actual,expected,d,w.tolerance,w.axes);result.update(name=name,inputs_unmutated=equal(inputs,before));checks.append(result);report['status']='running' if result['passed'] and result['inputs_unmutated'] else 'failed';save()
  assert result['passed'] and result['inputs_unmutated'],name
  del expected,before
  return actual
 set_seed(811);inputs=gen_inputs(d,w,'cuda:0',custom_inputs_fn=custom)
 first=compare('fresh_call',inputs);retained=snapshot(first)
 set_seed(812);fresh=gen_inputs(d,w,'cuda:0',custom_inputs_fn=custom)
 with torch.no_grad():
  for old,new in zip(inputs,fresh):
   if isinstance(old,torch.Tensor):old.set_(new)
   else:assert old==new,'non-tensor input changed'
 second=compare('same_tensor_object_new_storage',inputs);del second,fresh
 retained_ok=equal(first,retained);checks.append({'name':'retained_output_unchanged','passed':retained_ok});report['status']='running' if retained_ok else 'failed';save()
 assert retained_ok,'earlier output changed after later call'
 views=[v.view_as(v) if isinstance(v,torch.Tensor) else v for v in inputs]
 third=compare('fresh_view_same_storage',views);del third,views
 report['status']='passed';save()
 print('caller guards passed',task['id'],w.uuid)
if __name__=='__main__':main()
