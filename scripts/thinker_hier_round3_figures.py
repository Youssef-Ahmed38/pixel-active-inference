"""Aggregate frozen go/no-go results and render three measured-signature figures."""
import json,math
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from thinker_hier_collect import digest
from thinker_hier_round2 import save
from thinker_hier_round3_train import SEEDS,KINDS
from thinker_hier_round3_evaluate import VARIANTS,paths
from pai.thinker.hier_analysis import CAUSES,rate
from pai.thinker.hier3_tokens import truth_rows,load_split



def avg(values):
    v=np.array([x for x in values if x is not None],float)
    return dict(mean=float(v.mean()) if len(v) else None,std=float(v.std(ddof=1)) if len(v)>1 else 0.,values=v.tolist())

def aggregate():
    raw={kind:[json.loads(Path(paths(kind,seed)+'_final.json').read_text(encoding='utf8')) for seed in SEEDS] for kind in VARIANTS}
    summary={}
    for kind,runs in raw.items():
        summary[kind]={key:avg([r['summary'][key] for r in runs]) for key in
            ('nmi','mse','post20_mse','restricted_adaptation','mean_iterations','update_iterations','inner_past_error_improvement')}
        for key in ('cluster_accuracy','probe_accuracy'):
            summary[kind][key]=avg([r['summary'][key]['rate'] for r in runs])
            summary[kind][key]['per_seed_wilson']=[r['summary'][key] for r in runs]
        summary[kind]['timescales']={level:avg([r['summary']['timescales'][level]['mean'] for r in runs]) for level in ('level1','level2','level3')}
    delta=np.array(summary['hierarchical']['cluster_accuracy']['values'])-np.array(summary['flat']['cluster_accuracy']['values'])
    bound=float(stats.t.ppf(.975,2)*delta.std(ddof=1)/np.sqrt(3))
    comparison=dict(cluster_difference=float(delta.mean()),seed_std=float(delta.std(ddof=1)),
                    paired_seed_t95=[float(delta.mean()-bound),float(delta.mean()+bound)],
                    prediction_mse_ratio=summary['hierarchical']['mse']['mean']/summary['flat']['mse']['mean'])
    comparison['verdict']='GO' if delta.mean()>=.05 and comparison['paired_seed_t95'][0]>0 and comparison['prediction_mse_ratio']<1 else 'NO-GO'
    # Layout-cluster bootstrap: resample all five conditions for a layout jointly.
    layouts=sorted({r['seed'] for r in raw['hierarchical'][0]['records']})
    contributions=[]
    for layout in layouts:
        values=[]
        for i in range(3):
            h={r['index']:r for r in raw['hierarchical'][i]['records'] if r['seed']==layout}
            f={r['index']:r for r in raw['flat'][i]['records'] if r['seed']==layout}
            values.extend([(h[k]['label']==h[k]['cluster_cause'])-(f[k]['label']==f[k]['cluster_cause']) for k in h])
        contributions.append(float(np.mean(values)))
    rng=np.random.default_rng(93400)
    boot=np.mean(np.asarray(contributions)[rng.integers(len(layouts),size=(5000,len(layouts)))],1)
    comparison['layout_bootstrap95']=np.quantile(boot,[.025,.975]).tolist()
    thinking={}
    for kind,runs in raw.items():
        by={}
        for c in CAUSES:
            rs=[r for run in runs for r in run['records'] if r['label']==c]
            by[c]=dict(mean_iterations=float(np.mean([r['mean_iterations'] for r in rs])),
                       update_iterations=float(np.mean([r['update_iterations'] for r in rs])),
                       early_error=float(np.mean([r['early_error'] for r in rs if r['early_error'] is not None])) if c!='none' else None,
                       late_error=float(np.mean([r['late_error'] for r in rs if r['late_error'] is not None])) if c!='none' else None)
        x=[];y=[]
        for run in runs:
            for r in run['records']:
                for surprise,it in zip(r['surprise'],r['iterations']):
                    if it:x.append(surprise);y.append(it)
        corr=float(stats.spearmanr(x,y).statistic) if len(set(y))>1 else None
        thinking[kind]=dict(by_condition=by,spearman_surprise_iterations=corr)
    return raw,dict(models=summary,paired=comparison,thinking=thinking)


def figures(raw,out):
    colours=dict(hierarchical='#2768b2',flat='#d66b28',amortized='#438d63')
    fig,ax=plt.subplots(figsize=(8,4))
    labels=[];means=[];errors=[];colors=[]
    for kind in KINDS:
        levels=['level1'] if kind=='flat' else ['level1','level2','level3']
        for level in levels:
            a=out['models'][kind]['timescales'][level];labels.append(kind+'\n'+level);means.append(a['mean'] if a['mean'] is not None else float('nan'));errors.append(a['std']);colors.append(colours[kind])
    ax.bar(range(len(labels)),means,yerr=errors,color=colors,capsize=3);ax.set_xticks(range(len(labels)),labels,fontsize=8)
    ax.set_ylabel('Autocorrelation e-folding lag (steps)');ax.set_title('Measured timescales; update schedules are given')
    fig.tight_layout();fig.savefig('results/thinker_hier3_timescales.png',dpi=150);plt.close(fig)
    refs=json.loads(Path('results/thinker_hier3_references_final.json').read_text(encoding='utf8'))
    fig,axes=plt.subplots(2,2,figsize=(9,6))
    for ax,c in zip(axes.flat,CAUSES[1:]):
        for kind in KINDS:
            curves=[]
            for run in raw[kind]:
                arr=[r['post_curve'] for r in run['records'] if r['label']==c and len(r['post_curve'])==20]
                curves.append(np.mean(arr,0))
            curves=np.asarray(curves);mean=curves.mean(0);sd=curves.std(0,ddof=1)
            ax.plot(range(20),mean,label=kind,color=colours[kind]);ax.fill_between(range(20),mean-sd,mean+sd,color=colours[kind],alpha=.12)
        copied=[r['persistence_post_curve'] for r in refs['records'] if r['label']==c and len(r['persistence_post_curve'])==20]
        ax.plot(range(20),np.mean(copied,0),label='copy-current-senses',color='black',linestyle='--',alpha=.65)
        ax.set_title(c);ax.set_xlabel('Steps after logged onset');ax.set_ylabel('Normalized next-sense MSE')
    axes.flat[0].legend(fontsize=8);fig.suptitle('Event-aligned error: attenuation is not proof of neural repetition suppression')
    fig.tight_layout();fig.savefig('results/thinker_hier3_error_dynamics.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4));x=[];y=[]
    for run in raw['hierarchical']:
        for r in run['records']:
            for s,it in zip(r['surprise'],r['iterations']):
                if it:x.append(s);y.append(it)
    axes[0].scatter(np.log1p(x),y,s=3,alpha=.08);axes[0].set_xlabel('log(1 + own precision-weighted surprise)');axes[0].set_ylabel('Refinement iterations per update')
    cvals=out['thinking']['hierarchical']['by_condition'];axes[1].bar(range(5),[cvals[c]['update_iterations'] for c in CAUSES],color=colours['hierarchical'])
    axes[1].set_xticks(range(5),CAUSES,rotation=30,ha='right',fontsize=8);axes[1].set_ylabel('Mean iterations / update')
    fig.suptitle('Difficulty allocation uses a given surprise-dependent budget')
    fig.tight_layout();fig.savefig('results/thinker_hier3_thinking.png',dpi=150);plt.close(fig)


