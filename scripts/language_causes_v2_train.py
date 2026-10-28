"""Learn bilingual associations from the v1 physical episodes, now development data."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from pai.language.association import BeliefCoordinates,EvidenceLexicon,QuestionIntent
from pai.language.association_teacher import make_pairs,question_rows,scaffold
from pai.language.causal import ClauseLanguage
from pai.language.causal_teacher import PHRASES


def train(sources):
    rows=[r for path in sources for r in json.loads(Path(path).read_text(encoding='utf-8'))['episodes'] if r['mode']=='baseline']
    # Interleave classes in the learning curve; no final v2 data enters this file.
    conditions=('none','push','heavier_object','slippery_object','camera_shift')
    groups=[[r for r in rows if r['condition']==c] for c in conditions]
    rows=[r for batch in zip(*groups) for r in batch]
    coordinates=BeliefCoordinates.fit([r['belief'] for r in rows])
    result=dict(coordinates=dict(causes=coordinates.causes,features=coordinates.features),
                sources={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
                development_seeds=sorted({r['seed'] for r in rows}),episodes=len(rows),models={},diagnostics={})
    for language in ('english','arabic'):
        pairs=make_pairs(rows,language,coordinates)
        lexicon=EvidenceLexicon.fit(pairs)
        qtrain,qtest=question_rows(language)
        questions=QuestionIntent.fit(qtrain)
        qresults=[dict(text=t,expected=i,predicted=questions.predict(t)) for t,i in qtest]
        def mapping_score(model):
            return sum(model.phrase_by_coordinate.get(str(j))==PHRASES[language].get('cause:'+c)
                       for j,c in enumerate(coordinates.causes) if c!='unknown')
        curves=[]
        for count in (5,10,15,20,25):
            model=EvidenceLexicon.fit(pairs[:count])
            curves.append(dict(episodes=min(count,len(pairs)),correct_mappings=mapping_score(model),
                               learned=len(model.phrase_by_coordinate),numeric_links=len(model.numeric)))
        noisy=[]
        expected_numeric={('heavier_object','extra_mass_kg'),('camera_shift','camera_offset_mm'),
                          ('push','onset'),('camera_shift','onset')}
        def numeric_score(model):
            return sum(tuple(coordinates.features[m['feature']]) in expected_numeric for m in model.numeric)
        for seed in range(72810,72830):
            model=EvidenceLexicon.fit(make_pairs(rows,language,coordinates,corruption=.2,seed=seed))
            noisy.append(dict(seed=seed,wrong_labels=int(.2*len(rows)),correct_mappings=mapping_score(model),
                              learned=model.phrase_by_coordinate,numeric=model.numeric,numeric_correct=numeric_score(model)))
        shuffled=[]
        for seed in range(72900,72920):
            order=np.random.default_rng(seed).permutation(len(pairs))
            broken=[dict(r,posterior=pairs[int(j)]['posterior'],features=pairs[int(j)]['features']) for r,j in zip(pairs,order)]
            model=EvidenceLexicon.fit(broken)
            shuffled.append(dict(seed=seed,correct_mappings=mapping_score(model),numeric_links=len(model.numeric)))
        result['models'][language]=dict(lexicon=lexicon.__dict__,templates=ClauseLanguage.fit(scaffold(language)).templates,
                    questions=dict(vocabulary=questions.vocabulary,idf=questions.idf.tolist(),examples=questions.examples),
                    question_train=qtrain,question_test=qtest)
        result['diagnostics'][language]=dict(mapping_correct=mapping_score(lexicon),mapping_total=5,
                  curve=curves,noise_20_percent=noisy,question_results=qresults,
                  question_correct=sum(r['expected']==r['predicted'] for r in qresults),question_n=len(qresults),
                  shuffled_observation_control=shuffled,numeric_correct=numeric_score(lexicon))
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sources',nargs='+',default=['results/language_causes_development.json','results/language_causes_final.json'])
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report=train(args.sources)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({lang:{k:v for k,v in d.items() if k not in ('noise_20_percent','question_results')}
                      for lang,d in report['diagnostics'].items()},ensure_ascii=False,indent=2))
