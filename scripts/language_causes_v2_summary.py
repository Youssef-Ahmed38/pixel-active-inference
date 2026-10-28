"""Replay v2 evidence, verify scoring, and report Wilson intervals without tuning."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from pai.language.causal import EpisodeEvidence,ClauseLanguage
from language_causes_v2_eval import load_explainers,score_v2


def rate(k,n):
    if not n:
        return dict(k=k,n=n,rate=None,wilson95=None)
    z=1.959963984540054
    p=k/n
    den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return dict(k=int(k),n=int(n),rate=p,wilson95=[max(0.,mid-half),min(1.,mid+half)])


def summarize(report,checkpoint):
    if not report['complete']:
        raise ValueError('Incomplete final evaluation')
    conditions=('none','push','heavier_object','slippery_object','camera_shift')
    rows=report['episodes']
    keys=[(r['seed'],r['condition'],r['mode']) for r in rows]
    expected={(s,c,m) for s in report['seeds'] for c in conditions for m in ('baseline','english','arabic')}
    if len(keys)!=len(set(keys)) or set(keys)!=expected:
        raise ValueError('Missing or duplicate trials')
    explainers=load_explainers(checkpoint)
    bundles=0
    for row in rows:
        ev=EpisodeEvidence(**row['belief'])
        for lang,ex in explainers.items():
            o=row['outputs'][lang]
            if ex.explain(ev)!=o['explanation']:
                raise ValueError('Cannot reproduce explanation from evidence')
            if score_v2(o['explanation'],row['truth'],lang)!=o['score']:
                raise ValueError('Raw truth disagrees with score')
            for q in o['qa']:
                if ex.questions.predict(q['text'])!=q['predicted'] or ex.answer(q['text'],ev)!=q['answer']:
                    raise ValueError('Cannot reproduce question answer')
            bundles+=1
    out=dict(physical_trials=len(rows),layout_count=len(report['seeds']),bundles_replayed=bundles,
             conditions={},languages={},goals={},failures=[])
    def success(r):
        return bool(r['truth']['success'] and r['goal']==r['resolved_goal'])
    for c in conditions:
        out['conditions'][c]={m:rate(sum(success(r) for r in rows if r['condition']==c and r['mode']==m),
                                     sum(r['condition']==c and r['mode']==m for r in rows))
                              for m in ('baseline','english','arabic')}
    for goal in sorted({tuple(r['goal']) for r in rows}):
        out['goals'][str(goal)]={m:rate(sum(success(r) for r in rows if tuple(r['goal'])==goal and r['mode']==m),
                                       sum(tuple(r['goal'])==goal and r['mode']==m for r in rows))
                                for m in ('baseline','english','arabic')}
    baseline={(r['seed'],r['condition']):r for r in rows if r['mode']=='baseline'}
    for lang in explainers:
        rs=[r for r in rows if r['mode']==lang]
        os=[r['outputs'][lang] for r in rs]
        accepted=[o for o in os if not o['explanation']['abstained']]
        wording=report['question_wording'][lang]
        parameters={}
        for o in os:
            for p in o['score']['parameters']:
                parameters.setdefault(p['semantic'],[]).append(p)
        params={}
        for key,ps in parameters.items():
            eligible=[p for p in ps if p['event_matched']]
            errors=[p['error'] for p in eligible]
            params[key]=dict(issued=len(ps),unmatched_events=len(ps)-len(eligible),
                            accuracy=rate(sum(p['passed'] for p in eligible),len(eligible)),
                            mean_error=sum(errors)/len(errors) if errors else None)
        out['languages'][lang]=dict(n=len(rs),success=rate(sum(success(r) for r in rs),len(rs)),
            baseline_success=rate(sum(success(baseline[(r['seed'],r['condition'])]) for r in rs),len(rs)),
            paired_disagreements=sum(success(r)!=success(baseline[(r['seed'],r['condition'])]) for r in rs),
            cause_accuracy=rate(sum(o['score']['cause_correct'] for o in os),len(os)),
            coverage=rate(len(accepted),len(os)),precision=rate(sum(o['score']['cause_correct'] for o in accepted),len(accepted)),
            goal_accuracy=rate(sum(o['score']['goal_correct'] for o in os),len(os)),
            abstentions=len(os)-len(accepted),parameters=params,
            wording_accuracy=rate(sum(q['expected']==q['predicted'] for q in wording),len(wording)),
            wording_by_intent={str(i):rate(sum(q['expected']==q['predicted'] for q in wording if q['expected']==i),
                                           sum(q['expected']==i for q in wording))
                               for i in ('why','drop','cf:remove_push','cf:open_early',None)},
            wording=wording,
            end_to_end_why=rate(sum(q['predicted']=='why' and o['score']['cause_correct'] for o in os
                                    for q in o['qa'] if q['expected']=='why'),
                               sum(q['expected']=='why' for o in os for q in o['qa'])),
            cause_by_condition={c:rate(sum(r['outputs'][lang]['score']['cause_correct'] for r in rs if r['condition']==c),
                                       sum(r['condition']==c for r in rs)) for c in conditions})
        for r in rs:
            o=r['outputs'][lang]
            if not success(r) or not o['score']['cause_correct']:
                out['failures'].append(dict(seed=r['seed'],condition=r['condition'],goal=r['goal'],mode=lang,
                    success=success(r),agent_cause=o['explanation']['agent_cause'],
                    spoken_cause=o['score']['reported_cause'],abstained=o['explanation']['abstained']))
    return out


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    r=json.loads(args.input.read_text(encoding='utf-8'))
    checkpoint_path=Path(r['checkpoint'])
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()!=r['checkpoint_sha256']:
        raise ValueError('Changed frozen checkpoint')
    c=json.loads(checkpoint_path.read_text(encoding='utf-8'))
    out=summarize(r,c)
    legacy_path=Path('results/language_causes_final.json')
    if legacy_path.exists():
        legacy=json.loads(legacy_path.read_text(encoding='utf-8'))
        out['legacy_wording_baseline']={}
        for lang,questions in r['question_wording'].items():
            model=ClauseLanguage(legacy['learned_templates'][lang])
            checks=[dict(text=q['text'],expected=q['expected'],predicted=model.question(q['text'],r['seeds'][0]*100))
                    for q in questions]
            out['legacy_wording_baseline'][lang]=dict(
                accuracy=rate(sum(q['expected']==q['predicted'] for q in checks),len(checks)),questions=checks)
        out['legacy_source_sha256']=hashlib.sha256(legacy_path.read_bytes()).hexdigest()
    out.update(source=str(args.input),source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),command=[sys.executable,*sys.argv])
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({l:{k:v for k,v in d.items() if k in ('success','cause_accuracy','wording_accuracy')}
                      for l,d in out['languages'].items()},indent=2))
