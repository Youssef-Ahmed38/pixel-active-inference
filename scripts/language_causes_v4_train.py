"""Fit and freeze V4 surface composition and silence likelihoods on teaching data."""
import argparse
import hashlib
import json
from pathlib import Path
from pai.language.causal import ClauseLanguage
from pai.language.causal_v4 import CalibratedSilence,LearnedComposition
from pai.language.causal_v4_teacher import teacher,probability_rows
from pai.language.association_teacher import truth_cause


def build(old,rows):
    import copy
    out=copy.deepcopy(old)
    silence=CalibratedSilence.fit(rows,truth_cause)
    if any(r['n']<20 for r in silence.counts.values()) or len(silence.counts)!=5:
        raise ValueError('Require at least 20 new teaching episodes per condition')
    languages={}
    for lang in out['models']:
        t=teacher(lang)
        if lang=='arabic':
            strings=[*t['phrases'].values(),t['hedge'],*t['replacements'].values(),*(s for pair in t['pairs'] for s in pair),*(r['text'] for r in probability_rows(lang))]
            if any('?' in text or '\ufffd' in text for text in strings):
                raise ValueError('Arabic Unicode corruption; cannot freeze')
        model=LearnedComposition.fit(t['pairs'])
        languages[lang]=dict(composition=model.__dict__,phrases=t['phrases'],hedge=t['hedge'],replacements=t['replacements'],
                             probability_frame=ClauseLanguage.fit(probability_rows(lang)).templates['probability'],teacher_pairs=t['pairs'])
    out['v4']=dict(silence=silence.__dict__,languages=languages,confident_threshold=.95,minimum_claim=.5,
                   prior='balanced five-condition evaluation prior; independent Jeffreys Beta(0.5,0.5) likelihoods',
                   diagnostics={event:silence.diagnostics(event) for event in ('silent','explicit_none')})
    out['development_seeds']=sorted(set(out['development_seeds'])|{r['seed'] for r in rows})
    return out


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,data):
    text=json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    import re
    if re.search(r'[A-Za-z]:[\\/]|/Users/|/home/',text):
        raise ValueError('Absolute local path in results')
    temporary=Path(str(path)+'.tmp')
    temporary.write_text(text,encoding='utf8');temporary.replace(path)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--teaching',default='results/language_causes_v4_teaching.json')
    ap.add_argument('--old',default='results/language_causes_v3_checkpoint.json')
    ap.add_argument('--output',default='results/language_causes_v4_checkpoint.json')
    args=ap.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    data=json.loads(Path(args.teaching).read_text(encoding='utf8'))
    if not data['complete']:
        raise ValueError('Teaching collection incomplete')
    rows=[r for r in data['episodes'] if r['mode']=='baseline']
    old=json.loads(Path(args.old).read_text(encoding='utf8'))
    from language_causes_v2_eval import load_explainers
    from pai.language.causal import EpisodeEvidence
    models=load_explainers(old)
    replayed=0
    for row in rows:
        evidence=EpisodeEvidence(**row['belief'])
        for lang,model in models.items():
            if model.explain(evidence)!=row['outputs'][lang]['explanation']:
                raise ValueError('Development v3 teaching output does not replay')
            for q in row['outputs'][lang]['qa']:
                if model.answer(q['text'],evidence)!=q['answer']:
                    raise ValueError('Development question answer does not replay')
            replayed+=1
    out=build(old,rows)
    out['v4']['teaching_replay_bundles']=replayed
    out['v4']['calibration_episodes']=len(rows)
    out['v4']['teaching_seeds']=sorted({r['seed'] for r in rows})
    prior_seeds=set(); prior_files={}
    for p in Path('results').glob('language_causes*.json'):
        if 'v4' in p.name:
            continue
        prior=json.loads(p.read_text(encoding='utf8'))
        values=set(prior.get('seeds',[]))|set(prior.get('development_seeds',[]))
        if isinstance(prior.get('episodes'),list):
            values|={r['seed'] for r in prior['episodes'] if 'seed' in r}
        if values:
            prior_seeds|=values
            prior_files[str(p).replace('\\','/')]=dict(seeds=sorted(values),sha256=digest(p))
    if prior_seeds & (set(range(69020,69040))|set(out['v4']['teaching_seeds'])):
        raise ValueError('New seed range overlaps prior language results')
    out['v4']['freshness_audit']=prior_files
    out['v4']['sources']={p:digest(p) for p in (args.old,args.teaching)}
    out['v4']['frozen_sources']={str(p).replace('\\','/'):digest(p) for pattern in
                                ('pai/language/*.py','scripts/language_causes_v4*.py') for p in Path('.').glob(pattern)}
    out['v4']['frozen_runtime']={str(p).replace('\\','/'):digest(p) for pattern in ('pai/language/*.py','scripts/language_causes_v4_eval.py') for p in Path('.').glob(pattern)}
    out['v4']['frozen_runtime'].update({p:digest(p) for p in ('pai/agents/slice_agent.py','pai/causes/inference.py','pai/causes/credit.py','pai/envs/predicates.py','runs/slice_world_model/world_model.pt','scripts/language_causes_v2_eval.py','scripts/slice_eval.py','configs/tabletop.yaml')})
    out['v4']['final_plan']=dict(seeds=list(range(69020,69040)),disturbance_base=82000,modes=['baseline'],workers=[['none','push','heavier_object'],['slippery_object','camera_shift']],planning_samples=64,
                                rationale='Post-episode bilingual reporting; both checkpoints replay identical own evidence; no arm or instruction changes')
    save(args.output,out)
    print(json.dumps(out['v4']['diagnostics'],indent=2))
