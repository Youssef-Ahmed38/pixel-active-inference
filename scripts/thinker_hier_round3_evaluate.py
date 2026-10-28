"""Frozen development readouts and final scoring for the predictive hierarchy."""
import argparse,json
from pathlib import Path
import joblib
import numpy as np
import torch
from pai.thinker.predictive import PredictiveConfig
from pai.thinker.hier3_tokens import load_split,truth_rows
from pai.thinker.hier_analysis import CAUSES,cause,run_streams,fit_readouts,predict_readouts,episode_metrics,summarize
from thinker_hier_round3_train import KINDS,SEEDS,build,source_hashes
from thinker_hier_collect import digest
from thinker_hier_round2 import save
from thinker_hier_round3_collect import require_gate

VARIANTS=[*KINDS,'hierarchical_test_amortized']

def paths(kind,seed):
    return f'results/thinker_hier3_{kind}_s{seed}'

def model_for(kind,seed,device):
    trained='hierarchical' if kind=='hierarchical_test_amortized' else kind
    checkpoint=paths(trained,seed)+'.pt'
    payload=torch.load(checkpoint,map_location=device,weights_only=False)
    model=build(trained,PredictiveConfig(**payload['config'])).to(device)
    model.load_state_dict(payload['model']);model.eval()
    return model,payload,checkpoint

def score_rows(rows,results,truth,readouts,clean_level):
    predicted,probe,indices=predict_readouts(readouts,results);records=[]
    for r,out,p,q,k in zip(rows,results,predicted,probe,indices):
        tr=truth[r['index']]
        records.append(dict(index=r['index'],seed=tr['seed'],label=cause(tr),cluster_cause=CAUSES[int(p)],
             probe_cause=CAUSES[int(q)],cluster_index=int(k),success=tr['success'],
             **episode_metrics(out,tr,clean_level)))
    return records


def development(device):
    require_gate('gate_a');require_gate('gate_b')
    train_rows,training=load_split('train');dev_rows,dev=load_split('development');truth=truth_rows('development')
    assert len(training)==1000 and len(dev)==200
    labels=np.array([CAUSES.index(cause(truth[r['index']])) for r in dev_rows])
    for seed in SEEDS:
        for kind in VARIANTS:
            prefix=paths(kind,seed);output=Path(prefix+'_development.json')
            if output.exists():continue
            model,payload,checkpoint=model_for(kind,seed,device)
            norm=payload['normalization'];refine=False if kind=='hierarchical_test_amortized' else None
            tr=run_streams(model,training,norm,device,refine)
            dv=run_streams(model,dev,norm,device,refine)
            readouts=fit_readouts(tr,dv,labels,seed)
            clean_level=float(np.mean([out['error'].mean() for out,y in zip(dv,labels) if y==0]))
            joblib.dump(dict(readouts=readouts,clean_level=clean_level,checkpoint_sha256=digest(checkpoint)),prefix+'_readouts.joblib')
            records=score_rows(dev_rows,dv,truth,readouts,clean_level)
            save(output,dict(kind=kind,seed=seed,split='development',checkpoint=checkpoint,clean_level=clean_level,
                checkpoint_sha256=digest(checkpoint),records=records,summary=summarize(records)))
            print('development',kind,seed,'complete',flush=True)

def freeze():
    require_gate('gate_a');require_gate('gate_b')
    output=Path('results/thinker_hier3_freeze.json')
    if output.exists():raise FileExistsError(output)
    tests=json.loads(Path('results/thinker_hier3_tests_ready.json').read_text(encoding='utf8'))
    if tests['exit_code']!=0:raise ValueError('Required tests must pass before freeze')
    files=['results/thinker_hier3_plan.json','results/thinker_hier3_tests_ready.json','results/thinker_hier3_reference_calibration.json']+[f'results/thinker/checkpoints/real_c1000_hrm_s{s}.pt' for s in (0,1,2)]
    files += [p.as_posix() for split in ('calibration','development','train')
              for pattern in (f'thinker_hier3_{split}_manifest*.json',f'thinker_hier3_{split}_truth*.json')
              for p in Path('results').glob(pattern)]
    for seed in SEEDS:
        for kind in KINDS:files += [paths(kind,seed)+'.pt',paths(kind,seed)+'_training.json']
        for kind in VARIANTS:files += [paths(kind,seed)+'_readouts.joblib',paths(kind,seed)+'_development.json']
    assert all(Path(p).is_file() for p in files)
    save(output,dict(files={p:digest(p) for p in files},sources=source_hashes(),
                      final_seeds=list(range(99000,99040)),final_count=200,
                      rule='No final data opened during fitting; all parameters/readouts frozen',
                      command=['python',*__import__('sys').argv]))
    print('Frozen models and development-only mappings; final may now be collected',flush=True)

def final(device):
    frozen=json.loads(Path('results/thinker_hier3_freeze.json').read_text(encoding='utf8'))
    for p,h in frozen['files'].items():
        if digest(p)!=h:raise ValueError('Frozen model/readout changed: '+p)
    for p,h in frozen['sources'].items():
        if digest(p)!=h:raise ValueError('Frozen source changed: '+p)
    rows,streams=load_split('final');truth=truth_rows('final');assert len(streams)==200
    for row in rows:
        if digest(row['path'])!=row['sha256']:raise ValueError('Final sensory capture changed')
    seal_path=Path('results/thinker_hier3_final_capture_seal.json')
    seal={p.as_posix():digest(p) for pattern in ('thinker_hier3_final_manifest*.json','thinker_hier3_final_truth*.json')
          for p in Path('results').glob(pattern)}
    if seal_path.exists():
        if json.loads(seal_path.read_text(encoding='utf8'))['files']!=seal:raise ValueError('Final capture answer key changed')
    else:save(seal_path,dict(files=seal,rule='Seal immutable capture before the first final model score'))
    assert set(r['seed'] for r in truth.values())==set(frozen['final_seeds'])
    for seed in SEEDS:
        for kind in VARIANTS:
            prefix=paths(kind,seed);output=Path(prefix+'_final.json')
            if output.exists():continue
            model,payload,checkpoint=model_for(kind,seed,device)
            read=joblib.load(prefix+'_readouts.joblib');assert read['checkpoint_sha256']==digest(checkpoint)
            result=run_streams(model,streams,payload['normalization'],device,False if kind=='hierarchical_test_amortized' else None)
            records=score_rows(rows,result,truth,read['readouts'],read['clean_level'])
            save(output,dict(kind=kind,seed=seed,split='final',checkpoint=checkpoint,clean_level=read['clean_level'],
                checkpoint_sha256=digest(checkpoint),records=records,summary=summarize(records)))
            print('final',kind,seed,'complete',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=['development','freeze','final']);ap.add_argument('--device',default='cuda')
    args=ap.parse_args();torch.set_num_threads(1)
    if args.stage=='development':development(args.device)
    elif args.stage=='freeze':freeze()
    else:final(args.device)
