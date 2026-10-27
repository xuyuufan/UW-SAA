"""Verify the complete best-fixed-H matrix, published statistics and timing sample."""
from pathlib import Path
import csv,gzip,json,hashlib
from collections import defaultdict
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'results/best-h-main-20261028'
DRAWS,SEED=10000,202610195903
def read(name):
    with (P/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def close(a,b):
    assert abs(float(a)-float(b))<1e-10,(a,b)
def main():
    a=np.load(P/'data/validated_best_h_main_arrays.npz')
    values=a['relocations'];n=a['num_containers'];fps=a['fingerprints']
    solvers=list(a['solvers']);cids=list(a['condition_ids']);geos=list(a['geometries'])
    assert solvers==['EM','GP','ERI','EG','Chain','Minmax'] and values.shape==(6,41,21,40,2)
    assert list(a['arms'])==['BEST_FIXED_H','UWSAA'] and np.isfinite(values).all()
    assert np.equal(values,np.floor(values)).all() and (values>=0).all()
    assert n.shape==(21,40) and len(set(fps.flat))==840
    selected=json.loads((ROOT/'configs/best_fixed_h_by_solver.json').read_text())['selected_h_by_solver']
    assert selected==dict(zip(solvers,[20,20,10,20,20,20]))
    raw=json.loads((ROOT/'data/caserta_table2_840.json').read_text())
    dataset={}
    for e in raw:
        ranks={v:i+1 for i,v in enumerate(e['retrieval_sequence'])}
        normalized=sorted(tuple(ranks[v] for v in stack) for stack in e['initial_stacks'])
        payload=json.dumps({'num_stacks':e['num_stacks'],'max_tiers':e['max_tiers'],'rank_stacks':normalized},sort_keys=True,separators=(',',':')).encode('ascii')
        fp=hashlib.sha256(payload).hexdigest();dataset[fp]=e
    positions={str(fp):(gi,li) for gi,row in enumerate(fps) for li,fp in enumerate(row)}
    for fp,(gi,li) in positions.items():
        e=dataset[fp];assert e['parameter_group']==geos[gi] and len(e['retrieval_sequence'])==n[gi,li]
    seen=np.zeros(values.shape,dtype=bool)
    with gzip.open(P/'data/six_solver_best_h_main_endpoints.csv.gz','rt',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            si=solvers.index(r['solver']);ci=cids.index(r['condition_id']);gi,li=positions[r['layout_fingerprint']]
            ai=['BEST_FIXED_H','UWSAA'].index(r['arm']);key=(si,ci,gi,li,ai)
            assert not seen[key];seen[key]=True
            assert int(r['baseline_h'])==selected[r['solver']] and r['size']==geos[gi]
            assert int(r['relocations'])==values[key] and int(r['num_containers'])==n[gi,li]
            close(r['relocations_per_container'],values[key]/n[gi,li])
    assert seen.all()
    groups={f:[i for i,c in enumerate(cids) if c.split('|')[0]==f] for f in ['CLEAN','subset_shuffle','time_window_shuffle','delayed_positions','overlay_mix','regime_mix']}
    groups['ALL41']=list(range(41))
    rng=np.random.Generator(np.random.PCG64(SEED));weights=np.empty((21,DRAWS,40))
    for g in sorted(geos):
        idx=rng.integers(0,40,size=(DRAWS,40),dtype=np.int32)
        weights[geos.index(g)]=np.asarray([np.bincount(r,minlength=40) for r in idx])/40
    size_rows=read('data/size_solver_effects.csv');overall=read('data/overall_solver_effects.csv');numeric=0
    for metric in ['relocations','relocations_per_container']:
        tensor=values if metric=='relocations' else values/n[None,None,:,:,None]
        stats={}
        for group,ids in groups.items():
            layouts=tensor[:,ids].mean(axis=1);means=layouts.mean(axis=2)
            boot=np.einsum('gdl,sgla->sgda',weights,layouts,optimize=True)
            effect=(means[:,:,0]-means[:,:,1])/means[:,:,0]
            draws=(boot[:,:,:,0]-boot[:,:,:,1])/boot[:,:,:,0]
            gm=means.mean(axis=1);gb=boot.mean(axis=1)
            stats[group]=(means,effect,draws,gm,(gm[:,0]-gm[:,1])/gm[:,0],(gb[:,:,0]-gb[:,:,1])/gb[:,:,0])
        families=[stats[g] for g in ['subset_shuffle','time_window_shuffle','delayed_positions']]
        stats['PURE_BALANCED']=tuple(np.mean([p[i] for p in families],axis=0) for i in range(6))
        for r in size_rows:
            if r['metric']!=metric:continue
            si=solvers.index(r['solver']);gi=geos.index(r['source_group'])
            means,point,draws,*_=stats[r['group']];lo,hi=np.quantile(draws[si,gi],[.025,.975])
            for key,val in [('best_h_mean',means[si,gi,0]),('uwsaa_mean',means[si,gi,1]),
                            ('reduction_pct',point[si,gi]*100),('ci95_low_pct',lo*100),('ci95_high_pct',hi*100)]:close(r[key],val);numeric+=1
        for r in overall:
            if r['metric']!=metric:continue
            si=solvers.index(r['solver']);gm,p,b=stats[r['group']][3:];lo,hi=np.quantile(b[si],[.025,.975])
            for key,val in [('best_h_macro_mean',gm[si,0]),('uwsaa_macro_mean',gm[si,1]),('reduction_pct',p[si]*100),('ci95_low_pct',lo*100),('ci95_high_pct',hi*100)]:close(r[key],val);numeric+=1
    timing=defaultdict(list);seen=set();rule=json.loads((ROOT/'configs/gp_rule.json').read_text())
    with gzip.open(P/'data/current_six_solver_timing_sample.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line);si=solvers.index(r['solver']);ci=cids.index(r['condition_id']);gi,li=positions[r['layout_fingerprint']]
            ai=['UWSAA_BEST_FIXED_H','UWSAA'].index(r['arm']);key=(si,ci,gi,li,ai)
            assert key not in seen and li==0;seen.add(key)
            assert r['complete'] and r['relocations']==values[key] and len(r['decision_seconds'])==r['decisions']==r['relocations']
            close(sum(r['decision_seconds']),r['actor_seconds'])
            assert r['wall_seconds']>=r['actor_seconds']>=0
            if r['solver']=='GP':assert r['gp_model']['formula_sha256']==rule['formula_sha256']
            timing[r['solver'],ai].append(r)
    assert len(seen)==1764 and len(timing)==12 and all(len(x)==147 for x in timing.values())
    for row in read('tables/timing_current_six_solvers.csv'):
        for ai,prefix in [(0,'best_h_'),(1,'uwsaa_')]:
            records=timing[row['solver'],ai];decisions=[t for r in records for t in r['decision_seconds']]
            close(row[prefix+'decision_ms'],np.mean(decisions)*1000)
            close(row[prefix+'actor_ms_per_trajectory'],np.mean([r['actor_seconds'] for r in records])*1000)
            close(row[prefix+'wall_ms_per_trajectory'],np.mean([r['wall_seconds'] for r in records])*1000)
    newseen=set()
    with gzip.open(P/'data/eri_h10_missing_main.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line);ci=cids.index(r['condition_id']);gi,li=positions[r['layout_fingerprint']];key=(ci,gi,li)
            assert key not in newseen;newseen.add(key)
            assert r['complete'] and r['fixed_h']==10 and r['solver']=='ERI' and r['relocations']==values[2,ci,gi,li,0]
    assert len(newseen)==28560
    print(json.dumps({'status':'PASS_SIX_SOLVER_MAIN_AND_TIMING','main_endpoints':int(values.size),
                      'numeric_fields_recomputed':numeric,'bootstrap_draws':DRAWS,'new_eri_h10_endpoints':len(newseen),
                      'current_timing_trajectories':len(seen),'timing_scope':'147 sampled trajectories per arm/solver; not full MAIN timing'}))
if __name__=='__main__':main()
