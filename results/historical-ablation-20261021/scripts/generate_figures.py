"""Recreate historical ablation figures; all GP/MEAN6 rows retain legacy identity."""
from pathlib import Path
import argparse,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
SOLVERS=['EM','GP','ERI','EG','Chain','Minmax','MEAN6']
LABELS=['EM','GP (retired)','ERI','EG','Chain','Minmax','MEAN6 (historical)']
def read(name):
    with (ROOT/'data'/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def heat(rows,row_key,col_key,value_key,cols,names,out,stem,title):
    matrix=np.array([[float(next(r[value_key] for r in rows if r[row_key]==s and r[col_key]==c))*100 for c in cols] for s in SOLVERS])
    lim=max(1,np.ceil(np.max(np.abs(matrix))))
    fig,ax=plt.subplots(figsize=(8.8,3.8),layout='constrained')
    im=ax.imshow(matrix,cmap='PuOr_r',vmin=-lim,vmax=lim,aspect='auto')
    for i in range(len(SOLVERS)):
        for j in range(len(cols)):
            ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=8,
                    color='white' if abs(matrix[i,j])>lim*.65 else '#111111')
    ax.set_yticks(range(7),LABELS);ax.set_xticks(range(len(cols)),names,rotation=25,ha='right')
    ax.set_title(title,loc='left',fontsize=10);ax.tick_params(length=0)
    fig.colorbar(im,ax=ax,label='Relocation reduction (%)',shrink=.85)
    for ext in ['pdf','png','svg']:fig.savefig(out/(stem+'.'+ext),dpi=400,bbox_inches='tight')
    plt.close(fig)
def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,default=ROOT/'figures');args=p.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'serif','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    rows=[r for r in read('caserta_ablation_condition_effects.csv') if r['report_group']=='PURE_BALANCED' and r['metric']=='relocations']
    comps=['UWSAA_vs_NO_WINDOW_SHUFFLE_K8','UWSAA_vs_SINGLE_WINDOW_B2','UWSAA_vs_SINGLE_WINDOW_B5','UWSAA_vs_SINGLE_WINDOW_B10','UWSAA_vs_SINGLE_WINDOW_B20','UWSAA_vs_BEST_FIXED_H']
    heat(rows,'solver','comparison','estimate_relative_reduction',comps,['No shuffle','Window 2','Window 5','Window 10','Window 20','Best fixed H'],out,'historical_component_ablation','Historical component ablation: positive favors UWSAA; GP is retired')
    rows=read('scenario_ablation_primary_contrasts_all_solvers.csv');comps=list(dict.fromkeys(r['contrast_id'] for r in rows))
    heat(rows,'solver','contrast_id','estimate',comps,[c.replace('P1_','').replace('P2_','').replace('_',' ') for c in comps],out,'historical_scenario_ablation','Historical scenario controls: signed reduction for each registered contrast')
    rows=[r for r in read('accepted_ablation_primary_contrasts.csv') if r['stage']=='DETERMINISTIC_SINGLE' and r['metric']=='relocations']
    fig,ax=plt.subplots(figsize=(6.75,3.5),layout='constrained')
    for i,s in enumerate(SOLVERS):
        r=next(r for r in rows if r['solver']==s)
        ax.plot([float(r['ordinary_ci_low_percent']),float(r['ordinary_ci_high_percent'])],[i,i],c='#0072B2')
        ax.plot(float(r['relative_reduction_percent']),i,'o',c='#0072B2',ms=4)
    ax.set_yticks(range(7),LABELS);ax.set_ylim(6.6,-.6);ax.axvline(0,c='#888888',lw=.8,ls='--')
    ax.set_xlabel('Relocation reduction: UWSAA vs deterministic single (%)')
    ax.set_title('Historical deterministic-single comparison; ordinary paired 95% CI',loc='left',fontsize=9)
    for ext in ['pdf','png','svg']:fig.savefig(out/('historical_deterministic_single.'+ext),dpi=400,bbox_inches='tight')
    print('Historical figures written:',out)
if __name__=='__main__':main()
