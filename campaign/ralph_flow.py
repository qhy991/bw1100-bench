"""Three-hour engineering run: token accounting only, bounded search phase."""
import json,sys,time
from pathlib import Path
from typing import Any
from hmz.flows import Agent,Allowance,flow
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
@flow(budget=Allowance(hours=3),resumable=False)
def run(agents:tuple[Agent],task:str,state:dict[str,Any]|None=None)->None:
    (agent,)=agents
    while True:
        d=json.loads(Path('campaign/deadline.json').read_text())
        if time.time()>=d['search_stop_at_epoch']:return
        from campaign.statecard import render
        render()
        agent(task+'\nRead campaign/management/StateCard.json. Token usage is accounting only. Use only your own candidate/source history.',suppress=True)
        time.sleep(2)
