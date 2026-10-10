from pathlib import Path
import argparse,json,importlib.util,torch,time,statistics
R=Path('/work');p=argparse.ArgumentParser();p.add_argument('--emission',required=True);p.add_argument('--output',required=True);a=p.parse_args();rec=json.loads((R/a.emission).read_text());spec=importlib.util.spec_from_file_location('own_generated',R/rec['source']);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);fn=getattr(m,rec['entry']);prepared=torch.load(R/'campaign/development-inputs.pt',weights_only=False);rows=[]
for case in prepared:
 values={k:v.cuda() for k,v in case['tensors'].items()};out=fn(**values);torch.cuda.synchronize();outs=(out,) if isinstance(out,torch.Tensor) else tuple(out);observed={name:t.reshape(-1).cpu().tolist() for name,t in zip(rec['outputs'],outs)}
 samples=[]
 if case['case_id']=='primary':
  for _ in range(10):fn(**values)
  torch.cuda.synchronize()
  for _ in range(30):
   start=time.perf_counter();out=fn(**values);torch.cuda.synchronize();samples.append((time.perf_counter()-start)*1e6)
 rows.append({'case_id':case['case_id'],'outputs':observed,'after_inputs':{k:v.reshape(-1).cpu().tolist() for k,v in values.items()},'samples_us':samples});del values,out
 torch.cuda.synchronize()
torch.save(rows,R/a.output);print('device snapshots retained',len(rows))
