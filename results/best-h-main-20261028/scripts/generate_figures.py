from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
SOLVERS=['EM','GP','ERI','EG','Chain','Minmax'];HS=[20,20,10,20,20,20]
parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=ROOT/'figures');args=parser.parse_args();OUT=args.output_dir;OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.right':False,'axes.spines.top':False})
a=np.load(ROOT/'data/validated_best_h_main_arrays.npz');v=a['relocations'];geos=list(a['geometries']);cids=list(a['condition_ids']);labels=[g.replace('-','×') for g in geos]
family=[v[:,[i for i,c in enumerate(cids) if c.split('|')[0]==f]].mean(axis=1).mean(axis=2) for f in ['subset_shuffle','time_window_shuffle','delayed_positions']]
stats={'relocations':{'ALL41':(v.mean(axis=1).mean(axis=2),),'PURE_BALANCED':(np.mean(family,axis=0),)}}
def savefig(fig,stem):
    for ext in ['pdf','svg','png']:fig.savefig(OUT/(Path(stem).name+'.'+ext),dpi=450,bbox_inches='tight',facecolor='white')
    plt.close(fig)
def generate():
    # Wider full-range plots; the x span is physically more than twice the old figure.
    for group,tag in [('ALL41','main41'),('PURE_BALANCED','noisy')]:
        m=stats['relocations'][group][0]
        fig,axes=plt.subplots(2,3,figsize=(16,8),sharex=True,sharey=True,layout='constrained')
        y=np.arange(21)
        for si,(s,ax) in enumerate(zip(SOLVERS,axes.flat)):
            ax.hlines(y,m[si,:,0],m[si,:,1],color='#aaaaaa',lw=1.4)
            ax.plot(m[si,:,0],y,'o',color='#0072B2',ms=4,label=f'Best fixed H={HS[si]}')
            ax.plot(m[si,:,1],y,'D',color='#D55E00',ms=4,label='UWSAA')
            ax.set_yticks(y,labels);ax.set_ylim(20.7,-.7);ax.set_title(s,loc='left',weight='bold')
            ax.set_xlabel('Mean relocations');ax.grid(axis='x',alpha=.2)
            ax.legend(loc='lower right',bbox_to_anchor=(1,.10),fontsize=8,frameon=False)
            if si%3==0:ax.set_ylabel('Initial height × stacks')
        savefig(fig,f'figures/absolute_relocations_wide_{tag}')
    # Local ranges separate height groups, revealing small-instance differences.
    m=stats['relocations']['ALL41'][0];heights=[3,4,5,6,10]
    fig,axes=plt.subplots(6,5,figsize=(15,12),layout='constrained')
    for si,s in enumerate(SOLVERS):
        for hi,height in enumerate(heights):
            ix=[i for i,g in enumerate(geos) if int(g.split('-')[0])==height]
            ax=axes[si,hi];y=np.arange(len(ix));base=m[si,ix,0];uw=m[si,ix,1]
            ax.hlines(y,base,uw,color='#aaaaaa',lw=1.4)
            ax.plot(base,y,'o',c='#0072B2',ms=4)
            ax.plot(uw,y,'D',c='#D55E00',ms=4)
            ax.set_yticks(y,[labels[i] for i in ix],fontsize=8);ax.set_ylim(len(ix)-.5,-.5)
            lo=min(base.min(),uw.min());hi_v=max(base.max(),uw.max());pad=max((hi_v-lo)*.13,.2)
            ax.set_xlim(lo-pad,hi_v+pad);ax.grid(axis='x',alpha=.2);ax.tick_params(axis='x',labelsize=8)
            if si==0:ax.set_title(f'Initial height {height}',fontsize=11)
            if hi==0:ax.set_ylabel(s+f' (H={HS[si]})',fontsize=10)
            if si==5:ax.set_xlabel('Mean relocations',fontsize=9)
    handles=[plt.Line2D([],[],color='#0072B2',marker='o',ls='',label='Best fixed H'),
             plt.Line2D([],[],color='#D55E00',marker='D',ls='',label='UWSAA')]
    fig.legend(handles=handles,loc='outside upper center',ncol=2,frameon=False)
    savefig(fig,'figures/absolute_relocations_local_zoom_main41')

generate()
import csv
with (ROOT/'data/overall_solver_effects.csv').open(encoding='utf-8-sig') as f:
    rows=[r for r in csv.DictReader(f) if r['metric']=='relocations']
groups=['CLEAN','subset_shuffle','time_window_shuffle','delayed_positions','overlay_mix','regime_mix','PURE_BALANCED']
names=['CLEAN','SS','TW','DP','Overlay','Regime','SS/TW/DP']
colors=['#777777','#0072B2','#009E73','#E69F00','#CC79A7','#56B4E9','#D55E00']
fig,axes=plt.subplots(2,3,figsize=(9.5,5.5),sharex=True,layout='constrained')
for si,(solver,ax) in enumerate(zip(SOLVERS,axes.flat)):
    for yi,g in enumerate(groups):
        r=next(r for r in rows if r['solver']==solver and r['group']==g)
        ax.plot([float(r['ci95_low_pct']),float(r['ci95_high_pct'])],[yi,yi],c=colors[yi])
        ax.plot(float(r['reduction_pct']),yi,'o',c=colors[yi],ms=4)
    ax.set_yticks(range(7),names);ax.set_ylim(6.7,-.7);ax.axvline(0,c='#888888',ls='--',lw=.7)
    ax.set_title(solver+f': baseline H={HS[si]}',loc='left');ax.set_xlabel('Relocation reduction (%)');ax.grid(axis='x',alpha=.15)
savefig(fig,'family_effects_best_h_ci')
print("Figures written:",OUT)
