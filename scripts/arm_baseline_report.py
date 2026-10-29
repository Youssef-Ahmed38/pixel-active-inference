"""Summarize frozen paired results, learning curves and sample-efficiency provenance."""
import json
from pathlib import Path
import numpy as np

from pai.baselines.arm import CONDITIONS
from pai.baselines.statistics import rate,paired
from arm_baseline_plan import PREFIX,register,save,digest


def read(name):return json.loads(Path(PREFIX+name+'.json').read_text(encoding='utf8'))


def report():
    plan=register();full=read('final_full_agent')['records'];summary={};tests={}
    for method in plan['methods']:
        runs=[read('final_'+method+(f'_s{s}' if method in ('ppo','recurrent_ppo') else ''))['records'] for s in (plan['training_seeds'] if method in ('ppo','recurrent_ppo') else [None])]
        for run in runs:
            if len(run)!=100:raise ValueError('Incomplete final run')
        by={}
        for cond in ('overall',*CONDITIONS):
            rows=[[r for r in run if cond=='overall' or r['condition']==cond] for run in runs]
            rates=[rate(sum(r['success'] for r in rs),len(rs)) for rs in rows]
            by[cond]=dict(per_seed=rates,mean=float(np.mean([r['rate'] for r in rates])),std=float(np.std([r['rate'] for r in rates],ddof=1)) if len(rates)>1 else 0.,
                paired=[paired(rs,[r for r in full if cond=='overall' or r['condition']==cond]) for rs in rows],
                realized_disturbances=[sum(bool(r['realized_disturbances']) for r in rs) for rs in rows])
        summary[method]=by
    # Historical data provenance is limited: checkpoint has a data directory but
    # no capture hashes. Count the present corpus and disclose that limitation.
    import torch
    ckpt=torch.load(plan['config']['slice']['world_model'],map_location='cpu',weights_only=False)
    files=sorted(Path(ckpt['cfg']['world_model']['data_dir']).glob('ep_*.npz'));transitions=0
    for path in files:
        with np.load(path,allow_pickle=False) as d:transitions+=len(d['actions'])
    efficiency=dict(world_model=dict(current_corpus_episodes=len(files),current_corpus_decisions=transitions,
        control_50Hz_steps=transitions*5,optimizer_steps=ckpt['cfg']['world_model']['steps'],ensemble_members=ckpt['cfg']['world_model']['n_members'],
        batch_size=ckpt['cfg']['world_model']['batch_size'],provenance='Counts of current corpus at checkpoint data_dir; historical checkpoint lacks corpus hashes, so exact historical membership is not attested. Includes nominal scripted pick/place and exploration.'),learned={})
    for kind in ('ppo','recurrent_ppo'):
        efficiency['learned'][kind]=[read(f'{kind}_s{s}_training') for s in plan['training_seeds']]
    save(PREFIX+'summary.json',dict(results=summary,sample_efficiency={**efficiency,'learned':{k:[{x:r[x] for x in ('seed','environment_decisions','controlled_50Hz_steps','passive_settle_steps','reset_settle_steps','wall_s','stop','parameters')} for r in v] for k,v in efficiency['learned'].items()}}))
    lines=['\n## Frozen results\n','| Method | Overall | Clean | Push | Heavier | Slippery | Camera shift | Explanation |','|---|---:|---:|---:|---:|---:|---:|---|']
    for method,by in summary.items():
        cells=[]
        for c in ('overall',*CONDITIONS):
            v=by[c]
            if len(v['per_seed'])>1:cells.append(f"{100*v['mean']:.1f}% ± {100*v['std']:.1f}%")
            else:
                r=v['per_seed'][0];cells.append(f"{r['k']}/{r['n']} ({100*r['wilson95'][0]:.1f}–{100*r['wilson95'][1]:.1f}%)")
        lines.append('| '+method+' | '+' | '.join(cells)+' | '+('Own Bayesian cause/parameters/actions' if method in ('full_agent','no_adapt') else 'None')+' |')
    lines+=['','| Learned method / seed | Condition | Success | Wilson95 |','|---|---|---:|---|']
    for method in ('ppo','recurrent_ppo'):
        for j,seed in enumerate(plan['training_seeds']):
            for cond in ('overall',*CONDITIONS):
                r=summary[method][cond]['per_seed'][j]
                lines.append(f"| {method} / {seed} | {cond} | {r['k']}/{r['n']} | [{r['wilson95'][0]:.1%}, {r['wilson95'][1]:.1%}] |")
    lines+=['','Learned entries are mean ± sample SD over3 training seeds. Full per-run Wilson intervals and paired tests by condition are retained in arm_baselines_summary.json. These intervals describe episode uncertainty separately from training-seed variability. All per-episode failures remain in final JSON.','','| Candidate | Seed | Candidate-only success | Full-only success | Exact two-sided sign p |','|---|---:|---:|---:|---:|']
    for method,by in summary.items():
        if method=='full_agent':continue
        for seed,v in zip(plan['training_seeds'] if method in ('ppo','recurrent_ppo') else ['n/a'],by['overall']['paired']):
            lines.append(f"| {method} | {seed} | {v['candidate_only']} | {v['full_only']} | {v['two_sided_sign_p']:.6g} |")
    lines+=['','Paired tests are per training seed, with no pooling the same physical layout across three policies; multiple descriptive comparisons are not multiplicity-corrected. Assigned-condition scores include episodes that finished before a scheduled event. Realized-event counts are in the summary.','','| Agent | Cause correct overall |','|---|---|']
    for method in ('full_agent','no_adapt'):
        rows=read('final_'+method)['records'];r=rate(sum(x['condition']==x['inferred'] for x in rows),100)
        lines.append(f"| {method} | {r['k']}/100; Wilson95 [{r['wilson95'][0]:.1%}, {r['wilson95'][1]:.1%}] |")
    lines+=['','### Unedited own causal reports, next to scoring truth\n']
    for method in ('full_agent','no_adapt'):
        rows=read('final_'+method)['records']
        for cond in ('slippery_object','camera_shift'):
            r=next(r for r in rows if r['condition']==cond)
            lines += [f"{method}, first {cond} episode, layout {r['seed']}; scorer condition {cond}, true task success {r['success']}. Own report:",'```text',r['explanation'],'```']
    lines+=['','## Training and sample efficiency\n',f"The current nominal world-model corpus contains {transitions} decision transitions in {len(files)} episodes ({transitions*5} controlled50Hz steps). Checkpoint settings record8000 optimizer steps, batch512,5 ensemble members. This is not zero-shot or zero-training-data planning: data generation mixes70% scripted pick/place and30% exploration. Historical corpus hashes were not stored, so current corpus counts are a provenance-qualified estimate of the checkpoint experience. Reset settling is additional simulator work, not a learned transition.",'','| Baseline | Seed | Decision transitions | Controlled50Hz steps | Wall seconds | Stop |','|---|---:|---:|---:|---:|---|']
    for kind,runs in efficiency['learned'].items():
        for r in runs:lines.append(f"| {kind} | {r['seed']} | {r['environment_decisions']} | {r['controlled_50Hz_steps']} | {r['wall_s']:.1f} | {r['stop']} |")
    lines+=['','Before learning, Windows denied multiprocessing named pipes. A four-thread fallback worked; the final implementation uses independent CPU processes over ordinary anonymous stdio pipes. Physics workers load the exact slice disturbance function through AST extraction to avoid unnecessary Torch imports. Tests verify identical sampled disturbances. Up to3 independent training trials run concurrently, with4 one-thread simulation workers per trial (12 workers total on14 physical/20 logical CPU cores); PPO minibatches remain256 and GRU sequences32. Wall times are measured under this shared-resource schedule. Three1024-decision timing probes used no learning updates and separate layouts19000000+; all probe records and the pretraining I/O failure are retained.']
    lines+=['','Training curves (given shaped return and score-only true success) are retained per seed; passive and reset controls are counted separately in the sample-efficiency JSON. Four CPU simulation workers use one BLAS/OpenMP thread each; policy training runs on the laptop GPU.','','## Verdict and limitations\n',
        f"On these100 frozen episodes the full agent succeeds in {sum(r['success'] for r in full)}/100, compared with PPO {100*summary['ppo']['overall']['mean']:.1f}% ± {100*summary['ppo']['overall']['std']:.1f}% and recurrent PPO {100*summary['recurrent_ppo']['overall']['mean']:.1f}% ± {100*summary['recurrent_ppo']['overall']['std']:.1f}%. The paired table states the observed differences without assuming that the explicit-cause system must win. This comparison is conditional on the registered budgets and given priors; it does not isolate explicit causes from the arm's authored subgoals/demonstration-trained model or PPO reward/optimization choices. Low PPO clean-task success would indicate incomplete skill learning and cannot establish inherent superiority of cause inference. The full/no-adapt pair is the more direct adaptation ablation. Neither PPO has an explanation head; no model-based RL baseline was run.",
        '', '## Reproduce\n','```powershell',"$env:PYTHONPATH='.;scripts'","$env:OMP_NUM_THREADS='1'","$env:OPENBLAS_NUM_THREADS='1'",'.venv/Scripts/python.exe scripts/arm_baseline_plan.py','.venv/Scripts/python.exe scripts/arm_baseline_tests.py --basetemp data/pytest_arm_baselines_final_20261028 --output results/arm_baselines_tests_final.json','.venv/Scripts/python.exe scripts/arm_baseline_run.py','```',
        'The runner logs exact training/evaluation commands, freezes all checkpoints/sources before final evaluation, and resumes only completed episode keys/models. Use a fresh temporary test directory for any repeat. No final-based tuning, cloud compute or git state changes.']
    doc=Path('docs/ARM_BASELINES.md');marker='\n## Frozen results\n';old=doc.read_text(encoding='utf8')
    if marker in old:raise ValueError('Report already written; retain it')
    with doc.open('a',encoding='utf8') as handle:handle.write('\n'.join(lines)+'\n')
    # Standalone conventional plots: no raw step logs are committed.
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for kind,runs in efficiency['learned'].items():
        for i,r in enumerate(runs):
            x=[h['steps'] for h in r['history']]
            axes[0].plot(x,[h['rolling_return'] for h in r['history']],alpha=.65,label=f'{kind} seed{i}')
            axes[1].plot(x,[h['rolling_success'] for h in r['history']],alpha=.65)
    axes[0].set_ylabel('Rolling100 shaped return');axes[1].set_ylabel('Rolling100 score-only true success')
    for ax in axes:ax.set_xlabel('Environment decision transitions')
    axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(PREFIX+'learning_curves.png',dpi=150);plt.close(fig)
    with doc.open('a',encoding='utf8') as handle:handle.write('\n![Learning curves](../results/arm_baselines_learning_curves.png)\n')


if __name__=='__main__':report()
