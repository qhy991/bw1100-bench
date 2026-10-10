"""Confirm one fixed, source-bound nominee inside the original three-hour budget."""
from pathlib import Path
import copy,hashlib,json,math,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from campaign.completion import validate


def source_bound(root, outcome):
    bindings=dict(outcome.get('cake_artifact_bindings',{}))
    bindings[outcome['candidate_source']]=outcome['source_sha256']
    return all((root/name).is_file() and hashlib.sha256((root/name).read_bytes()).hexdigest()==value
               for name,value in bindings.items())


def main():
    os.chdir(ROOT)
    deadline=json.loads((ROOT/'campaign/deadline.json').read_text())
    intake=json.loads((ROOT/'campaign/intake.json').read_text())
    plan=json.loads((ROOT/intake['plan']).read_text())
    folder=ROOT/'campaign/final-confirmation';folder.mkdir(exist_ok=False)
    outcomes=[json.loads(p.read_text()) for p in (ROOT/'campaign/evaluations').glob('*/outcome.json')]
    eligible=[x for x in outcomes if x.get('status')=='accepted'
              and x['completed_at_epoch']<=deadline['search_stop_at_epoch']]
    stale=[x['id'] for x in eligible if not source_bound(ROOT,x)]
    eligible=[x for x in eligible if x['id'] not in stale]
    nominee=max(eligible,key=lambda x:x['conservative_geomean']) if eligible else None
    result={'scope':'one fixed nominee; independent paired/A-A confirmation inside total3h',
            'status':'missing_qualified_nominee','nominated_id':None,
            'stale_artifacts_excluded':stale,'budget_stop_epoch':deadline['stop_at_epoch']}
    try:
        if nominee:
            validate(ROOT,{'tasks':[nominee['handoff_row']]},plan)
            result['nominated_id']=nominee['id']
            (folder/'nomination.json').write_text(json.dumps(nominee,indent=2))
            timeout=min(900,int(deadline['stop_at_epoch']-time.time())-45)
            if timeout<10:
                result['status']='insufficient_remaining_budget'
            else:
                latest=deadline['stop_at_epoch']-timeout-30
                command=['python3','scripts/hcu_run.py','--image',intake['image'],'--device',str(plan['hcu']),
                    '--timeout',str(timeout),'--queue-timeout','300','--latest-start-epoch',str(latest),
                    '--receipt','campaign/final-confirmation/admission.json','--','env','TRITON_F32_DEFAULT=ieee',
                    'FLA_TRIL_PRECISION=ieee','AITER_JIT_DIR=/work/.local/aiter-jit-cache',
                    'TRITON_CACHE_DIR=/work/.local/triton-cache','python3','campaign/paired_wall.py',
                    '--candidate','/work/'+nominee['candidate_source'],
                    '--gate','/work/'+nominee['handoff_row']['correctness_report'],
                    '--samples','30','--output','campaign/final-confirmation/latency.json']
                with (folder/'execution.log').open('x') as log:
                    code=subprocess.call(command,stdout=log,stderr=subprocess.STDOUT)
                result.update(status='confirmation_failed',exit_code=code)
                if code==0:
                    if not source_bound(ROOT,nominee):raise ValueError('nominee artifacts changed during final confirmation')
                    report=json.loads((folder/'latency.json').read_text())
                    terminal=json.loads((folder/'admission-terminal.json').read_text())
                    row=copy.deepcopy(nominee['handoff_row'])
                    row['latency_report']='campaign/final-confirmation/latency.json'
                    # Reuse the original full UUID, source, baseline, caller and A/A gate.
                    validate(ROOT,{'tasks':[row]},plan)
                    ratios=[min(c['baseline_us'][order]/c['candidate_us'][order]
                                for order in ['forward_median','reverse_median']) for c in report['rows']]
                    released=terminal['exit_code']==0 and not terminal['container_still_running'] and terminal['after_vram']=='0%' and not terminal['after_kfd_visible']
                    timely=terminal['completed_at']<=deadline['stop_at_epoch']
                    result.update(status='confirmed' if released and timely else 'not_confirmed',
                        conservative_geomean=math.exp(sum(math.log(x) for x in ratios)/len(ratios)),
                        rows=len(ratios),paired_AA_gate_passed=True,released=released,completed_within_budget=timely)
    except (KeyError,ValueError,OSError,subprocess.SubprocessError) as error:
        result.update(status='confirmation_failed',error=type(error).__name__+': '+str(error))
    result['completed_node_epoch']=time.time()
    with (folder/'result.json').open('x') as f:json.dump(result,f,indent=2)
    print(result,flush=True)


if __name__=='__main__':main()
