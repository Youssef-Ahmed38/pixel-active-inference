"""Aggregate frozen go/no-go results and render three measured-signature figures."""
import json,math
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from thinker_hier_collect import save,digest
from thinker_hier_train import SEEDS,KINDS
from thinker_hier_evaluate import VARIANTS,paths
from pai.thinker.hier_analysis import CAUSES,truth_rows,rate
from pai.thinker.hier_data import load_split


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
    fig.tight_layout();fig.savefig('results/thinker_hier_timescales.png',dpi=150);plt.close(fig)
    refs=json.loads(Path('results/thinker_hier_references_final.json').read_text(encoding='utf8'))
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
    fig.tight_layout();fig.savefig('results/thinker_hier_error_dynamics.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4));x=[];y=[]
    for run in raw['hierarchical']:
        for r in run['records']:
            for s,it in zip(r['surprise'],r['iterations']):
                if it:x.append(s);y.append(it)
    axes[0].scatter(np.log1p(x),y,s=3,alpha=.08);axes[0].set_xlabel('log(1 + own precision-weighted surprise)');axes[0].set_ylabel('Refinement iterations per update')
    cvals=out['thinking']['hierarchical']['by_condition'];axes[1].bar(range(5),[cvals[c]['update_iterations'] for c in CAUSES],color=colours['hierarchical'])
    axes[1].set_xticks(range(5),CAUSES,rotation=30,ha='right',fontsize=8);axes[1].set_ylabel('Mean iterations / update')
    fig.suptitle('Difficulty allocation uses a given surprise-dependent budget')
    fig.tight_layout();fig.savefig('results/thinker_hier_thinking.png',dpi=150);plt.close(fig)


def document(raw,out):
    plan=json.loads(Path('results/thinker_hier_plan.json').read_text(encoding='utf8'))
    refs=json.loads(Path('results/thinker_hier_references_final.json').read_text(encoding='utf8'))
    def fmt(a,percent=False):
        factor=100 if percent else 1
        return f"{a['mean']*factor:.4f} ± {a['std']*factor:.4f}" if a['mean'] is not None else 'undefined'
    lines=['# A predictive hierarchy through RGB and proprioception: registered go/no-go','',
    '## Design and truth boundary','',
    'This is an offline predictive/cause-discovery experiment on a fixed arm behavior policy, not an end-to-end replacement of the arm controller. L1 is a 48-unit GRU predicting the next 19-dimensional sensory vector. Learned per-channel Gaussian precision weights its error. L2 contains eight categorical logits (softmax code) plus eight continuous coordinates, with an amortized update every four steps or on a large own prediction error. The iterative version minimizes the completed previous transition’s prediction error plus a weak proximity-to-prior penalty using latent-only gradients (1–5 updates). It then predicts the next observation; it never refines using that future target. A first-order straight-through path trains amortization and prediction without second-order differentiation. L3 is a learned 16-unit episode/goal-context GRU updated every 16 steps and reset per episode; it conditions L2. L3 is not a learned planner.','',
    'Flat: one 75-unit GRU. Amortized: exactly the hierarchical architecture, one inference pass, no latent gradient refinement. All see the same sensed features, actions, holding belief and phase; the latter two are the existing controller’s internal states, not cause labels. All train with the same Gaussian next-observation objective, optimizer steps, batch indices per seed and truncated windows. Hierarchical and amortized have 31,974 parameters; flat has 32,084 (0.34% difference). Training uses 1,200 AdamW steps, batch 16, windows 24, learning rate 0.001, fixed last checkpoint. No validation-label hyperparameter search or final-set tuning was performed. Truncated-window zero initialization may disadvantage long episode context; a negative result is specific to this small design and budget.','',
    'Learned: predictive/recurrent weights, L2 code/continuous representation, L3 context dynamics, and channel precisions, from senses only. Given: architecture, update periods, refinement learning rate/cap and surprise-budget rule, motor policy, object identities/colors/shapes, nominal camera calibration, and RGB geometric decoder. The decoder uses rendered RGB only for object positions, with proprioception/touch for the gripper and occluded grasp geometry; no `obs[state]`, simulator object positions, predicates, current camera pose or events enter it. Its estimates are noisy and its colored-object/known-geometry priors are authored. These are not learned visual representations. A sensory step is one 0.1-second planning interval, not every 50 Hz physics/sensor tick. The main predictors do not consume Bayesian residuals, HRM posteriors, labels, event onsets, row IDs or seeds. Raw proprioceptive joint streams are saved, although the compared input uses gripper position/force/opening, perceived target and mean scene position, RGB mean/std, action, holding and phase.','',
    'A single opt-in `sensory` collection argument was added to `run_episode`; default behavior is unchanged. Collection disables cause-driven adaptations and uses the existing world model/MPPI with samples=16, horizon=6, one optimizer iteration, an 80-step cap and no monitored settle tail. A passive monitor records reference residuals only; labels cannot influence motor adaptation. The exploratory controller is given, and curiosity/EFE acquisition is not learned by this new module. Student surprise allocates inference effort; training samples remain uniform.','',
    '## Data, seeds and registration','',
    'Pilot: seed 89800 under five conditions, 0/5 task successes, retained with noisy perception and incomplete manipulation. Training: 90000–90199, 200 layouts × five conditions = 1,000 episodes. Development: 91000–91039, 40 × five = 200. Frozen final: 92000–92039, 40 × five = 200. These are distinct layout ranges; the same layout is reused across five conditions within a split. Disturbance RNG is `layout_split_base + 10000 + condition_index*1000 + layout_index`, recorded by the collector; same-condition disturbance draws and physical episodes do not cross splits. Models use training seeds 93000, 93001, 93002. Every negative result/variant is retained.','',
    'Sensory `.npz` files contain only own streams, and separate truth/reference files hold events and old-world-model residuals for scoring. The loader rejects unexpected fields; training opens no truth/reference files. Normalization and PCA16/KMeans5 fitting use training data only, without labels. Cluster-to-cause Hungarian mapping and a fixed-C logistic probe are fitted on development labels only and frozen before final data collection. Episode embeddings are the mean/std of causal states over the full sensed trajectory; no logged onset selects an embedding. Prediction windows and adaptation alignment use event onsets only in scoring. Late pushes can leave fewer than 20 samples before the 80-step collection cap: such cases stay in overall prediction/discovery scores, but are excluded uniformly from complete-post20 MSE and the capped adaptation metric; their partial curves and available lengths remain in the records. This measurement correction was made before final collection and does not alter the registered primary go/no-go metrics.','',
    'Registered GO rule: hierarchy must gain at least five percentage points in mean cluster accuracy over flat, its paired training-seed Student-t 95% lower bound must exceed zero, and its mean overall prediction MSE must be lower. Otherwise NO-GO. The rule was registered before model training; the final set was not used for selecting models, thresholds, mappings, or stopping steps.','',
    '## Go/no-go results','',
    'Mean ± sample standard deviation over three training seeds. Individual seed count metrics retain Wilson 95% intervals in the JSON and table below; no false independence is claimed across training seeds or repeated layouts. MSE is in train-standardized sensory coordinates, averaged per episode; smaller is better.','',
    '| Model | Cluster accuracy % | NMI | Linear probe % | Overall MSE | First 20 after onset MSE | Capped return-to-clean steps |',
    '|---|---:|---:|---:|---:|---:|---:|']
    for kind in VARIANTS:
        s=out['models'][kind]
        lines.append(f"| {kind} | {fmt(s['cluster_accuracy'],True)} | {fmt(s['nmi'])} | {fmt(s['probe_accuracy'],True)} | {fmt(s['mse'])} | {fmt(s['post20_mse'])} | {fmt(s['restricted_adaptation'])} |")
    lines.extend(['','| Model | Seed | Cluster correct / n [Wilson 95%] | Probe correct / n [Wilson 95%] |','|---|---:|---|---|'])
    for kind,runs in raw.items():
        for run in runs:
            cells=[]
            for key in ('cluster_accuracy','probe_accuracy'):
                r=run['summary'][key];lo,hi=r['wilson95'];cells.append(f"{r['k']}/{r['n']} [{100*lo:.1f}–{100*hi:.1f}%]")
            lines.append(f"| {kind} | {run['seed']} | {cells[0]} | {cells[1]} |")
    lines.extend(['','| Model | Seed | Initial detectable spikes | Censored returns / disturbed [Wilson 95%] |','|---|---:|---:|---|'])
    for kind,runs in raw.items():
        for run in runs:
            rs=[r for r in run['records'] if r['complete_post20']]
            r=rate(sum(x['censored'] for x in rs),len(rs));lo,hi=r['wilson95']
            lines.append(f"| {kind} | {run['seed']} | {sum(x['initial_spike'] for x in rs)} | {r['k']}/{r['n']} [{100*lo:.1f}–{100*hi:.1f}%] |")
    lines.extend(['','Categorical codes are not forced to match the five causes. Dominant-code counts below can reveal collapse even when a post-hoc continuous-feature probe separates classes.','',
                  '| Model | Seed | Dominant-code counts (0–7) | Mean categorical entropy (nats) |','|---|---:|---|---:|'])
    for kind,runs in raw.items():
        if kind=='flat':continue
        for run in runs:
            lines.append(f"| {kind} | {run['seed']} | {run['summary']['code_usage']} | {run['summary']['code_entropy']:.5f} |")
    p=out['paired']
    lines.extend(['',f"Hierarchy minus flat cluster accuracy: **{100*p['cluster_difference']:.2f} pp**, paired training-seed t95 [{100*p['paired_seed_t95'][0]:.2f}, {100*p['paired_seed_t95'][1]:.2f}] pp; layout-cluster bootstrap95 [{100*p['layout_bootstrap95'][0]:.2f}, {100*p['layout_bootstrap95'][1]:.2f}] pp (5,000 draws, seed 93400). Hierarchy/flat prediction-MSE ratio: **{p['prediction_mse_ratio']:.4f}**.",'',
    'Adaptation is the endpoint of the first completed three-consecutive-step moving mean below that model’s development clean mean, within 20 steps after a logged onset. On fully observed 20-step windows, missing returns are censored and contribute 20 to the capped comparative score; counts/initial-spike flags are preserved per episode. Payload/friction begin at episode start and may not become observable until a later grasp, so an early “return” does not prove adaptation. No detectable initial spike is explicitly marked.','',
    '## References using teacher supervision or authored cause likelihoods','',
    f"Bayesian reference: {refs['bayes']['k']}/{refs['bayes']['n']}, Wilson95 {refs['bayes']['wilson95']}. It uses its authored likelihoods with residual-noise calibration fitted on training clean labels (reference only). Existing c1000 HRM imitation checkpoints are evaluated on the same final perceived residuals: {[r['rate'] for r in refs['hrm']]}. They used cause labels/Bayesian teacher during their original training and are references, not unsupervised competitors. The present RGB decoder and behavior budget differ from c1000, so these are distribution-shift results, not a replacement of its original 89.7% result. The original 96.0% agreement vs transformer 95.8% does not establish a hierarchy advantage.",'',
    f"Copy-current-senses prediction reference: overall normalized MSE {refs['persistence_mse']:.6f}, post-onset MSE {refs['persistence_post20']:.6f}. This reference has no learned dynamics or cause representation and helps identify attenuation explained by sensory persistence alone.",'',
    '## Measured signatures and their limits','',
    '| Model | L1 ACF time (steps) | L2 ACF time | L3 ACF time | Iterations / sensory step | Iterations / update | Past-error reduction / update |','|---|---:|---:|---:|---:|---:|---:|'])
    for kind in VARIANTS:
        s=out['models'][kind];ts=s['timescales']
        lines.append(f"| {kind} | {fmt(ts['level1'])} | {fmt(ts['level2']) if kind!='flat' else 'same hidden state'} | {fmt(ts['level3']) if kind!='flat' else 'same hidden state'} | {fmt(s['mean_iterations'])} | {fmt(s['update_iterations'])} | {fmt(s['inner_past_error_improvement'])} |")
    lines.extend(['','Autocorrelation time is the first lag with normalized state autocorrelation at most exp(-1), capped at min(40, episode length/2). Undefined constant traces and capped traces are recorded. Higher-level update schedules are authored, so slower measured states are not evidence that biological timescales emerged.','',
    '![Measured timescales](../results/thinker_hier_timescales.png)','',
    '| Condition | Iterations/update, hierarchy | Early error (first 5) | Late error (15–20) |','|---|---:|---:|---:|'])
    for c,r in out['thinking']['hierarchical']['by_condition'].items():
        early='—' if r['early_error'] is None else f"{r['early_error']:.5f}";late='—' if r['late_error'] is None else f"{r['late_error']:.5f}"
        lines.append(f"| {c} | {r['update_iterations']:.3f} | {early} | {late} |")
    lines.extend(['','![Event-aligned prediction errors](../results/thinker_hier_error_dynamics.png)','',
    'The event-aligned figure measures persistence/attenuation, not neural mismatch negativity or a demonstrated repetition-suppression mechanism. Each episode contains one injected event; a short push ends after three planning steps, and attenuation can reflect that external ending, the given motor policy, or simple sensory autocorrelation. Persistent mass/friction conditions can remain unexposed if the arm never grasps.','',
    f"Spearman correlation of own surprise and iterations on hierarchy update steps: {out['thinking']['hierarchical']['spearman_surprise_iterations']}. This relationship is partly imposed by the given budget rule, not a learned halting head. The separately trained amortized model and the same hierarchy weights with refinement disabled at test are both reported; past-error minimization is not itself proof of improved future prediction or discovered causes.",'',
    '![Thinking allocation](../results/thinker_hier_thinking.png)','',
    '## Failures, data qualification, and scope','',
    'Low-quality perception, grasp failures, variable goal phases, and the authored collection controller can shape representations. Hungarian/probe accuracy is observational separability, not proof of causal identifiability or intervention discovery. The categorical code may collapse or encode pose/task progress rather than disturbance type; no labels were used to repair it. No local predictive-coding weight updates, learned visual encoder, new action planner, autonomous curiosity, or closed-loop student-control improvement is demonstrated. These remain outside this experiment. Per-episode errors, censored adaptations, code-independent readouts and all nine training histories are retained.','',
    '| Split | Condition | Episodes | Task successes | Holding belief present | Logged grasp present |','|---|---|---:|---:|---:|---:|'])
    for split in ('train','development','final'):
        truths=truth_rows(split);rows,streams=load_split(split);lookup={r['index']:s for r,s in zip(rows,streams)}
        for c in CAUSES:
            selected=[r for r in truths.values() if r['assigned_condition']==c]
            held=sum(bool(lookup[r['index']]['context'][:,0].any()) for r in selected)
            grasp=sum(any(e['type']=='grasp' for e in r['events']) for r in selected)
            lines.append(f"| {split} | {c} | {len(selected)} | {sum(r['success'] for r in selected)} | {held} | {grasp} |")
    lines.extend(['','## Reproduce','',
    'Use the project `.venv/Scripts/python.exe`, repository root, `PYTHONPATH=.;scripts`, `OMP_NUM_THREADS=1`. All compute is local. Commands run:','',
    '```powershell',
    '.venv/Scripts/python.exe scripts/thinker_hier_collect.py --split pilot',
    '.venv/Scripts/python.exe scripts/thinker_hier_train.py --register',
    '.venv/Scripts/python.exe scripts/thinker_hier_collection.py --split train',
    '.venv/Scripts/python.exe scripts/thinker_hier_collection.py --split development',
    '.venv/Scripts/python.exe scripts/thinker_hier_training.py',
    '.venv/Scripts/python.exe scripts/thinker_hier_references.py calibration',
    '.venv/Scripts/python.exe scripts/thinker_hier_evaluate.py development',
    '.venv/Scripts/python.exe scripts/thinker_hier_evaluate.py freeze',
    '.venv/Scripts/python.exe scripts/thinker_hier_collection.py --split final',
    '.venv/Scripts/python.exe scripts/thinker_hier_evaluate.py final',
    '.venv/Scripts/python.exe scripts/thinker_hier_references.py final',
    '.venv/Scripts/python.exe scripts/thinker_hier_report.py',
    '.venv/Scripts/python.exe scripts/thinker_hier_tests.py --basetemp data/pytest_thinker_hier_final_20261028 --output results/thinker_hier_tests_final.json',
    '```','',
    'Collection resumes completed episode keys without repeating them. Frozen source/model/readout hashes are checked before final scoring. Existing result files must be retained; do not rerun final data to select a more favorable design. JSON paths are repository-relative. Windows tests use inherited permissions only for newly created test temporary directories and never delete existing directories.','',
    '## Verdict','',
    f"**{p['verdict']}** under the registered rule. This particular three-level predictor changes final cluster accuracy by {100*p['cluster_difference']:.2f} percentage points relative to the same-size flat GRU, with paired training-seed 95% interval [{100*p['paired_seed_t95'][0]:.2f}, {100*p['paired_seed_t95'][1]:.2f}] pp and prediction-MSE ratio {p['prediction_mse_ratio']:.4f}. The reported signatures are measurements under given update/effort schedules, not independent evidence of a brain mechanism. This result applies to the stated RGB decoder, data quality, and training budget; it neither establishes nor rules out a broader advantage of hierarchical active inference.",''])
    Path('docs/THINKER_HIERARCHY.md').write_text('\n'.join(lines),encoding='utf8')

if __name__=='__main__':
    raw,out=aggregate();figures(raw,out);document(raw,out)
    out['files']={str(p).replace('\\','/'):digest(p) for p in Path('results').glob('thinker_hier_*_final.json')}
    save('results/thinker_hier_summary.json',out)
    print(json.dumps(out['paired'],indent=2))
