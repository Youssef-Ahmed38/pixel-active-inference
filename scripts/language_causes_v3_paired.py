"""Paired v2 vs v3 explainers on the same saved v3 final episodes (no new simulation).

The explainer runs after the episode on the saved belief, so both frozen checkpoints can be
replayed on identical evidence; only the reporting layer differs.
"""
import argparse
from collections import Counter
import json
from math import comb
from pathlib import Path
from pai.language.causal import EpisodeEvidence
from language_causes_v2_eval import load_explainers,score_v2


def sign_test(a,b):
    n,k=a+b,min(a,b)
    return 1. if not n else min(1.,2*sum(comb(n,i) for i in range(k+1))/2**n)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,default=Path('results/language_causes_v3_final.json'))
    ap.add_argument('--old',type=Path,default=Path('results/language_causes_v2_checkpoint.json'))
    ap.add_argument('--new',type=Path,default=Path('results/language_causes_v3_checkpoint.json'))
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    report=json.loads(args.input.read_text(encoding='utf-8'))
    models={name:load_explainers(json.loads(p.read_text(encoding='utf-8'))) for name,p in (('v2',args.old),('v3',args.new))}
    out=dict(input=str(args.input),languages={})
    for lang in ('english','arabic'):
        rows=[r for r in report['episodes'] if r['mode']==lang]
        correct={}
        res={}
        for name,ex in models.items():
            by,n,abst=Counter(),Counter(),0
            correct[name]=[]
            for r in rows:
                e=ex[lang].explain(EpisodeEvidence(**r['belief']))
                s=score_v2(e,r['truth'],lang)
                by[r['condition']]+=s['cause_correct']; n[r['condition']]+=1; abst+=e['abstained']
                correct[name].append(bool(s['cause_correct']))
            res[name]=dict(cause_correct=sum(by.values()),n=len(rows),abstentions=abst,
                           by_condition={c:[by[c],n[c]] for c in n})
        only_new=sum(b and not a for a,b in zip(correct['v2'],correct['v3']))
        only_old=sum(a and not b for a,b in zip(correct['v2'],correct['v3']))
        res['only_v3_correct'],res['only_v2_correct']=only_new,only_old
        res['sign_test_p']=sign_test(only_new,only_old)
        out['languages'][lang]=res
    args.output.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))
