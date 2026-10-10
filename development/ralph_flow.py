"""Three-hour engineering run: token accounting only, bounded search phase."""
import json,sys,time
from pathlib import Path
from typing import Any
from hmz.flows import Agent,Allowance,flow
sys.path.insert(0,str(Path(__file__).resolve().parent))
from development_stop import stopped
@flow(budget=Allowance(hours=3),resumable=True)
def run(agents:tuple[Agent],task:str,state:dict[str,Any]|None=None)->None:
    (agent,)=agents
    binding=json.loads(Path('campaign/development-binding.json').read_text())
    while True:
        if stopped(binding):return
        d=json.loads(Path('campaign/deadline.json').read_text())
        if time.time()>=d['search_stop_at_epoch']:return
        agent(task+'\nRead your own campaign/evaluations outcomes and generated source. Token usage is accounting only.',suppress=True)
        if stopped(binding):return
        request = Path('campaign/SEARCH_CLOSE.json')
        if request.exists():
            text = request.read_text()
            ready = Path('campaign/author-round-terminal.json')
            temporary = ready.with_suffix('.json.tmp')
            temporary.write_text(json.dumps({'request': text, 'node_epoch': time.time()}))
            temporary.replace(ready)
            # No further agent call starts while the owner validates closure.
            while time.time() < d['search_stop_at_epoch']:
                if stopped(binding):return
                try:
                    decision = json.loads(Path('campaign/search-close-owner.json').read_text())
                except (OSError, ValueError):
                    decision = {}
                if decision.get('request') == text:
                    if decision.get('decision') == 'approved':
                        return
                    if decision.get('decision') == 'refused':
                        break
                time.sleep(1)
        time.sleep(2)
