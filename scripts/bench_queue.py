"""Finite Bench-only author sequencer; the existing HCU gateway exclusively owns device admission."""
from pathlib import Path
import argparse,collections,fcntl,json,os,subprocess,sys,time
ROOT = None

def read(path):return json.loads(path.read_text())
def atomic(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def release_audit(root, preexisting_json=()):
 jobs={};issues=[];declared=set();seen=set();preexisting=set(preexisting_json)
 def consume(p):
  if str(p.relative_to(root)) in preexisting or p in seen:return
  seen.add(p)
  if p.is_symlink():issues.append({'file':str(p.relative_to(root)),'reason':'symlink_not_followed'});return
  try:d=read(p)
  except (ValueError,OSError):issues.append({'file':str(p.relative_to(root)),'reason':'unreadable_or_partial_record'});return
  if not isinstance(d,dict):return
  refs=d.get('admission_receipts',[])
  if isinstance(refs,list):declared.update(x for x in refs if isinstance(x,str))
  if d.get('schema')!='bw1100-bench.hcu-admission.v1':return
  jid=d.get('job_id')
  if not isinstance(jid,str) or not jid:issues.append({'file':str(p.relative_to(root)),'reason':'missing_job_id'});return
  row=jobs.setdefault(jid,{'terminals':[],'records':[]});row['records'].append(str(p.relative_to(root)))
  if all(k in d for k in ['completed_at','exit_code','container_still_running','after_vram','after_kfd_visible']):row['terminals'].append(d)
 def walk_error(error):issues.append({'file':str(error.filename),'reason':'unreadable_directory'})
 for directory,dirs,files in os.walk(root,onerror=walk_error):
  dirs[:]=[x for x in dirs if x not in ['.git','.deps','.data','triton-cache','aiter-jit-cache','__pycache__']]
  for name in files:
   if name.endswith('.json'):consume(Path(directory)/name)
 for ref in sorted(declared):
  try:p=(root/ref).resolve();p.relative_to(root.resolve())
  except (ValueError,OSError):issues.append({'file':ref,'reason':'declared_receipt_outside_root'});continue
  if not p.is_file():issues.append({'file':ref,'reason':'declared_receipt_missing'})
  elif str(p.relative_to(root)) in preexisting:issues.append({'file':ref,'reason':'declared_receipt_is_preexisting_source'})
  else:consume(p)
 for jid,row in jobs.items():
  if not row['terminals']:issues.append({'job_id':jid,'reason':'missing_terminal'})
  elif any(d['container_still_running'] is not False or d['after_vram']!='0%' or d['after_kfd_visible'] is not False for d in row['terminals']):issues.append({'job_id':jid,'reason':'release_not_confirmed_by_terminal'})
 return {'jobs':list(jobs),'issues':issues,'release_records_clean':not issues}

def done_observation(path):
 if not path.exists():return None,'missing_DONE'
 try:
  value=read(path)
  if not isinstance(value,dict):return None,'DONE_is_not_object'
  return value,None
 except (ValueError,OSError) as error:return None,type(error).__name__

def run(root, allocation):
 global ROOT
 ROOT = root.resolve()
 ROOT.mkdir(parents=True, exist_ok=True)
 with (ROOT/'owner.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not list(ROOT.glob('launch-*.json')) and not (ROOT/'LAUNCH.json').exists(),'Existing launch: inspect, never restart'
  spec=read(allocation);pending=list(spec['runs']);children={};finished=[];attention=[];blocked={}
  hcus=sorted({r['hcu'] for r in pending});capacity=spec['capacity_per_hcu']
  if not pending or len({r['frozen_workspace'] for r in pending})!=len(pending) or len({Path(r['frozen_workspace']).name for r in pending})!=len(pending):raise ValueError('Expected unique nonempty host allocation')
  if not 1<=capacity<=3 or any(h not in range(1,7) for h in hcus):raise ValueError('Unsupported author capacity or HCU')
  if type(spec['start_window_seconds']) is not int or spec['start_window_seconds']<=0:raise ValueError('A positive frozen start window is required')
  if any(r['host']!=spec['host'] for r in pending):raise ValueError('Foreign host allocation')
  for r in pending:
   if read(Path(r['root'])/'campaign/binding.json')!={k:v for k,v in r.items() if k!='prepared_commit'}:raise ValueError('Host subset differs from prepared binding')
   if subprocess.check_output(['git','-C',r['root'],'rev-parse','HEAD'],text=True).strip()!=r['prepared_commit']:raise ValueError('Prepared source changed')
   subprocess.run([sys.executable,'campaign/launch.py','--check'],cwd=r['root'],check=True)
  started=time.time();stop=started+spec['start_window_seconds']
  with (ROOT/'LAUNCH.json').open('x') as f:json.dump({'owner_pid':os.getpid(),'node_epoch':started,'maximum_runs':len(pending),'no_restarts':True,'host':spec['host'],'maximum_authors':len(hcus)*capacity},f,indent=2)
  def state(phase):
   atomic(ROOT/'STATUS.json',{'phase':phase,'owner_pid':os.getpid(),'node_epoch':time.time(),'started_node_epoch':started,'stop_new_at_node_epoch':stop,'host':spec['host'],'capacity_per_hcu':capacity,'running':[dict(r,pid=p.pid) for r,p in children.values()],'finished':finished,'pending':[r['frozen_workspace'] for r in pending],'release_attention':attention,'blocked_hcus':blocked,'scope':'author capacity only; HCU gateway owns all device locks and checks'})
  while True:
   for key,(r,p) in list(children.items()):
    code=p.poll()
    if code is None:continue
    root=Path(r['root']);audit=release_audit(root,read(ROOT/r['source_json_inventory']));done,done_error=done_observation(root/'campaign/DONE.json');row=dict(r,exit_code=code,ended_node_epoch=time.time(),done=done,done_read_error=done_error,release_audit=audit)
    finished.append(row);del children[key]
    if audit['issues'] or done_error or code:attention.append({'id':Path(r['frozen_workspace']).name,'hcu':r['hcu'],'root':r['root'],**audit})
   # Only canonical terminal receipts confirm release. An empty container list
   # cannot discharge missing, unreadable or contradictory release evidence.
   blocked={}
   for item in attention:
    if not item['release_records_clean']:
     blocked.setdefault(item['hcu'],[]).append({'id':item['id'],'issues':item['issues']})
   counts=collections.Counter(r['hcu'] for r,p in children.values())
   if time.time()<stop:
    for h in hcus:
     while counts[h]<capacity and h not in blocked:
      i=next((i for i,r in enumerate(pending) if r['hcu']==h),None)
      if i is None:break
      r=dict(pending.pop(i));r['hcu']=h;root=Path(r['root']);record=ROOT/('launch-'+Path(r['frozen_workspace']).name+'.json')
      assert not (root/'campaign/deadline.json').exists() and not record.exists()
      assert subprocess.check_output(['git','-C',str(root/'.deps/cake-ir'),'rev-parse','HEAD'],text=True).strip()==r['compiler_commit']
      assert read(root/'campaign/groups/c.json')['hcu']==h
      inventory=ROOT/'intake'/ (Path(r['frozen_workspace']).name+'-source-json.json');inventory.parent.mkdir(exist_ok=True)
      source_json=subprocess.check_output(['git','-C',str(root),'ls-files','--','*.json'],text=True).splitlines()
      with inventory.open('x') as f:json.dump(source_json,f)
      r['source_json_inventory']=str(inventory.relative_to(ROOT))
      with record.open('x') as f:json.dump(dict(r,state='reserved_before_intake',node_epoch=time.time()),f,indent=2)
      command=[sys.executable,'campaign/launch.py']
      try:
       with (root/'campaign/logs/owner-launch.log').open('x') as log:p=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
      except OSError as e:
       finished.append(dict(r,exit_code=None,launch_error=str(e),done=None));attention.append({'id':Path(r['frozen_workspace']).name,'hcu':h,'root':r['root'],'jobs':[],'issues':[{'reason':'launch_failed','error':str(e)}],'release_records_clean':True});continue
      atomic(record,dict(r,pid=p.pid,node_epoch=time.time()));children[Path(r['frozen_workspace']).name]=(r,p);counts[h]+=1
      print('LAUNCHED',Path(r['frozen_workspace']).name,'HCU',h,'PID',p.pid,flush=True)
   if not pending and not children:
    state('terminal_with_release_attention' if attention else 'terminal');return
   if time.time()>=stop and not children:state('start_window_expired');return
   state('running');time.sleep(20)
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--allocation',required=True,type=Path)
 parser.add_argument('--state-directory',required=True,type=Path)
 args=parser.parse_args()
 run(args.state_directory,args.allocation)
