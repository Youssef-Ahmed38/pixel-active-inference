"""Read-only checks and prefix/control comparisons for frozen arm likelihoods."""
import argparse
import copy
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import numpy as np
from pai.causes.inference import infer_cause
from pai.causes.learned import KNOWN,LearnedArmLikelihood,extract_features,peak_fit
from scripts.arm_causes import load,save,sha,sources,history,actual_cause,metrics,wilson


def audit(args):
    data=load(args.input);frozen=load(args.model);scores=load(args.scores)
    assert data['complete'] and not data['source_changes_during_run']
    assert frozen['frozen_sources']==sources()
    assert scores['input_sha256']==sha(args.input) and scores['model_sha256']==sha(args.model)
    m=LearnedArmLikelihood.from_dict(frozen['terminal']['model'])
    dt=data['config']['env']['control_dt']*data['config']['slice']['action_repeat']
    counts={'episodes':0,'baseline_reports_replayed':0,'learned_reports_replayed':0,'poisoned_truth_checks':0}
    for r,scored in zip(data['episodes'],scores['rows']):
        assert (r['seed'],r['scheduled_condition'])==(scored['seed'],scored['scheduled_condition'])
        assert set(r['own'])=={'history','trigger_steps'}
        hist=history(r)
        before=m.infer(hist,dt,bool(r['own']['trigger_steps']))
        assert before.best==scored['new_cause']
        assert all(np.isclose(before.posterior[c],scored['new_posterior'][c],rtol=1e-7,atol=1e-10) for c in before.posterior)
        old=peak_fit(hist,dt) if r['own']['trigger_steps'] else None
        assert (old.best if old else 'none')==scored['old_cause']
        if old:
            assert all(np.isclose(old.posterior[c],r['baseline']['posterior'][c],rtol=1e-5,atol=1e-7) for c in old.posterior)
            counts['baseline_reports_replayed']+=1
        poisoned=copy.copy(r)
        poisoned['scoring']={'cause':'unknown','events':[{'mass':999,'offset':[99,99,99]}]}
        poisoned['scheduled_condition']='unknown';poisoned['baseline']={'posterior':{'unknown':1.}}
        after=m.infer(history(poisoned),dt,bool(poisoned['own']['trigger_steps']))
        assert before.posterior==after.posterior and before.params==after.params
        if data['controller']=='learned':
            assert r['controller_terminal']['cause']==before.best
            assert all(np.isclose(r['controller_terminal']['posterior'][c],before.posterior[c],rtol=1e-5,atol=1e-7) for c in before.posterior)
        counts['episodes']+=1;counts['learned_reports_replayed']+=1;counts['poisoned_truth_checks']+=1
    return {'passed':True,'counts':counts,'input_sha256':sha(args.input),'model_sha256':sha(args.model),
            'checks':['same frozen sources','baseline and new diagnoses on identical sensor histories',
                      'baseline posterior replay','learned posterior replay','truth and label mutation invariance',
                      'native learned terminal agrees with saved own-evidence replay']}


def online(args):
    data=load(args.input);frozen=load(args.model)
    assert sources()==frozen['frozen_sources']
    m=LearnedArmLikelihood.from_dict(frozen['online']['model'])
    repeat=data['config']['slice']['action_repeat'];dt=data['config']['env']['control_dt']*repeat
    rows=[];episodes=[]
    for r in data['episodes']:
        hist=history(r);issued=[]
        for i,e in enumerate(hist):
            if e.t not in r['own']['trigger_steps']:continue
            prefix=hist[:i+1]
            old=infer_cause(prefix[-20:],dt);new=m.infer(prefix,dt,True)
            true_events=[ev for ev in r['scoring']['disturbances'] if ev['t']<=(e.t+1)*repeat]
            row={'seed':r['seed'],'condition':r['scheduled_condition'],'t':e.t,
                 'truth':actual_cause(true_events),'old':old.best,'new':new.best,
                 'old_confidence':max(old.posterior.values()),'new_confidence':max(new.posterior.values())}
            rows.append(row);issued.append(row)
        episodes.append({'seed':r['seed'],'condition':r['scheduled_condition'],'truth':r['scoring']['cause'],
                         'old':issued[-1]['old'] if issued else 'none',
                         'new':issued[-1]['new'] if issued else 'none','reports':len(issued)})
    summary={stage:{method:metrics([r['truth'] for r in rs],[r[method] for r in rs]) for method in ('old','new')}
             for stage,rs in [('at_alarms',rows),('last_report_per_episode',episodes)]}
    return {'kind':'prefix_diagnostic_replay','input_sha256':sha(args.input),'model_sha256':sha(args.model),
            'regime':data['regime'],'controller':data['controller'],'summary':summary,
            'silent_episodes':sum(r['reports']==0 for r in episodes),'rows':rows,'episodes':episodes,
            'note':'At-alarm results condition on the actual monitor trigger schedule; prefix inputs exclude future evidence.'}


def control(args):
    old=load(args.input);new=load(args.other)
    assert old['complete'] and new['complete'] and old['controller']=='baseline' and new['controller']=='learned'
    assert old['config']==new['config'] and old['calibration_sha256']==new['calibration_sha256']
    index={(r['seed'],r['scheduled_condition']):r for r in new['episodes']}
    old_keys={(r['seed'],r['scheduled_condition']) for r in old['episodes']}
    assert len(index)==len(new['episodes']) and set(index)<=old_keys
    rows=[]
    for r in old['episodes']:
        if (r['seed'],r['scheduled_condition']) not in index:continue
        q=index[(r['seed'],r['scheduled_condition'])]
        rows.append({'seed':r['seed'],'condition':r['scheduled_condition'],
                     'baseline_success':r['scoring']['success'],'learned_success':q['scoring']['success'],
                     'baseline_cause':r['scoring']['cause'],'learned_cause':q['scoring']['cause']})
    summaries={}
    for c in (*KNOWN,'all'):
        rs=[r for r in rows if c=='all' or r['condition']==c];n=len(rs)
        summaries[c]={m:{'correct':sum(r[m+'_success'] for r in rs),'n':n,
                         'wilson95':wilson(sum(r[m+'_success'] for r in rs),n)} for m in ('baseline','learned')}
    rng=np.random.default_rng(870001);seeds=sorted({r['seed'] for r in rows});deltas=[]
    for _ in range(5000):
        selected=rng.choice(seeds,len(seeds));rs=[r for s in selected for r in rows if r['seed']==s]
        deltas.append(np.mean([r['learned_success'] for r in rs])-np.mean([r['baseline_success'] for r in rs]))
    return {'kind':'matched_controller_goal_comparison','regime':old['regime'],'summary':summaries,'rows':rows,
            'baseline_episodes_outside_matched_subset':len(old['episodes'])-len(rows),
            'paired':{'learned_only_success':sum(r['learned_success'] and not r['baseline_success'] for r in rows),
                      'baseline_only_success':sum(r['baseline_success'] and not r['learned_success'] for r in rows),
                      'layout_bootstrap95':np.quantile(deltas,[.025,.975]).tolist(),'bootstrap_seed':870001},
            'input_sha256':sha(args.input),'other_sha256':sha(args.other),
            'note':'Controllers share initial seeds/schedules, but their actions differ; classifier comparisons separately use identical traces.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('audit','online','control'))
    p.add_argument('--input',type=Path,required=True);p.add_argument('--other',type=Path)
    p.add_argument('--model',type=Path);p.add_argument('--scores',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    result={'audit':audit,'online':online,'control':control}[args.mode](args)
    save(args.output,result)
    print({k:v for k,v in result.items() if k in ('passed','counts','paired','silent_episodes')})
