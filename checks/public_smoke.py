from __future__ import annotations
from pathlib import Path
import sys
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from environment.noise_aware_forecast import PublicObservation
from environment.public_frame import PublicDecisionFrame
from environment.public_history import PublicHistoryTracker, attach_history_summary
from uwsaa.entrypoint import SOLVERS,load_gp_rule,load_runtime,make_policy

rows=((1,21,22,22,0),(2,3,4,5,0),(6,7,8,9,0),(10,11,12,13,0),(14,15,16,0,0),(17,18,19,0,0),(20,22,22,0,0))
ranks=np.asarray(rows,dtype=np.int32); occupied=ranks!=0
legal=np.asarray((False,True,True,True,True,True,True),dtype=np.bool_)
obs=PublicObservation(rank_tensor=ranks,occupied_mask=occupied,legal_destination_mask=legal,history_hits=(),q_prior=5183/9920,q_t=5183/9920)
frame=PublicDecisionFrame(observation=obs,legal_destinations=(1,2,3,4,5,6),max_tiers=5,crane_position=0,crane_time=0.0,retrieval_count=0,relocation_count=0)
frame=attach_history_summary(frame,PublicHistoryTracker().summary(frame))
runtime=load_runtime(); gp=load_gp_rule(); results={}
for solver in SOLVERS:
 results[solver]={}
 for method in ('UWSAA','NATIVE20'):
  action,trace=make_policy(solver,method,runtime,gp).select(frame)
  if action not in frame.legal_destinations: raise AssertionError((solver,method,action))
  results[solver][method]={'action':action,'trace':trace}
payload={'check':'cpu_public_one_frame_smoke','status':'PASS','fixture':'public-25-item-one-frame','solvers':list(SOLVERS),'methods':['UWSAA','NATIVE20'],'scope':'one decision per solver/method; no training, trajectory, or performance evaluation','results':results}
((ROOT/'checks/public_smoke_result.json').write_text(json.dumps(payload,indent=2,sort_keys=True,default=str)+'\n',encoding='utf-8'))
print(json.dumps(payload,sort_keys=True,default=str))
