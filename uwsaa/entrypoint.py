"""One-frame public UWSAA and Native20 entrypoints."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping
from uwsaa.fast import UWSAAFastSelector
from baseline.native import SOLVERS,FixedHorizonActor,SingleStepPolicy
from baseline import eri_eg,chain,minmax
from baseline.em import select_em_destination
from baseline.eri_eg_public import select_eri20_destination
from environment.public_frame import truncate_public_frame,public_frame_to_legacy_state
from baseline.gp import select_destination as gp_select_destination

ROOT=Path(__file__).resolve().parents[1]

def load_runtime()->dict[str,Any]:
    return {"truncate_public_frame":truncate_public_frame,"public_frame_to_legacy_state":public_frame_to_legacy_state,"em_select":select_em_destination,"eri_select":select_eri20_destination,"eri_eg":eri_eg}

def load_gp_rule()->dict[str,Any]:
    from baseline.gp_expression import Node
    from baseline.gp import select_destination
    payload=json.loads((ROOT/"configs/gp_accepted_rule.json").read_text(encoding="utf-8"))
    tree=Node.from_prefix(tuple(int(x) for x in payload["selected_best_prefix"]))
    if tree.expression()!=payload["selected_best_expression"]: raise ValueError("copied GP prefix does not match its accepted expression")
    def selector(state,legal_destinations,*,max_tiers): return int(select_destination(tree,state,legal_destinations,max_tiers=max_tiers))
    return {"selector":selector,"tree":tree,"identity":payload["model_identity"]}

def make_policy(solver:str,method:str,runtime:Mapping[str,Any],gp_rule:Mapping[str,Any]):
    if solver not in SOLVERS: raise ValueError(f"unknown solver {solver!r}; expected one of {SOLVERS}")
    if method=="UWSAA": return UWSAAFastSelector(solver,runtime,gp_rule)
    if method!="NATIVE20": raise ValueError("method must be UWSAA or NATIVE20")
    if solver in SOLVERS[:4]: return FixedHorizonActor(solver,20,runtime,gp_rule)
    proposer=chain.propose if solver=="Chain" else minmax.propose
    return SingleStepPolicy(solver,proposer)

def select_once(solver,method,frame,*,runtime=None,gp_rule=None):
    return make_policy(solver,method,load_runtime() if runtime is None else runtime,load_gp_rule() if gp_rule is None else gp_rule).select(frame)
