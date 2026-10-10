from pathlib import Path
import argparse,json,subprocess,os,time
from development_stop import stopped, halt, device_failure_is_infrastructure
R=Path(__file__).resolve().parents[1];os.chdir(R);p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--id',required=True);p.add_argument('--phase',choices=['search','confirmation'],default='search');a=p.parse_args();b=json.loads((R/'campaign/development-binding.json').read_text());d=json.loads((R/'campaign/deadline.json').read_text());folder=R/'campaign/evaluations'/a.id
if stopped(b):raise SystemExit('batch STOP: no new evaluations')
os.environ['BWBENCH_STOP_FILE']=b['batch_stop_file']
if a.phase=='search' and (R/'campaign/SEARCH_CLOSE.json').exists():raise SystemExit('search-close request pending; no new device work')
code=subprocess.call(['bash','scripts/dtk.sh','cpu',b['image'],'python3','campaign/development_evaluate.py','--candidate',a.candidate,'--id',a.id,'--phase','emit','--result-phase',a.phase])
if code or not (folder/'emission.json').exists():raise SystemExit(code)
stop=d['search_stop_at_epoch'] if a.phase=='search' else d['stop_at_epoch'];timeout=min(600,int(stop-time.time())-30);assert timeout>=10
if stopped(b):raise SystemExit('batch STOP: no device admission')
if a.phase=='search' and (R/'campaign/SEARCH_CLOSE.json').exists():raise SystemExit('search-close request pending; no device admission')
cmd=['python3','scripts/hcu_run.py','--image',b['image'],'--device',str(b['hcu']),'--timeout',str(timeout),'--queue-timeout','300','--latest-start-epoch',str(stop-timeout-30),'--receipt',str((folder/'admission.json').relative_to(R)),'--','env','TRITON_F32_DEFAULT=ieee','TRITON_CACHE_DIR=/work/.local/triton-cache','python3','campaign/development_device.py','--emission',str((folder/'emission.json').relative_to(R)),'--output',str((folder/'observed.pt').relative_to(R))]
with (folder/'device.log').open('x') as log:code=subprocess.call(cmd,stdout=log,stderr=subprocess.STDOUT)
if code:
 reason=device_failure_is_infrastructure(folder)
 if reason:halt(b,reason,str(folder.relative_to(R)))
 (folder/'outcome.json').write_text(json.dumps({'id':a.id,'phase':a.phase,'status':'device_failed','exit_code':code,'scope':'development only'},indent=2));raise SystemExit(code)
code=subprocess.call(['bash','scripts/dtk.sh','cpu',b['image'],'python3','campaign/development_evaluate.py','--candidate',a.candidate,'--id',a.id,'--phase','compare','--result-phase',a.phase]);raise SystemExit(code)
