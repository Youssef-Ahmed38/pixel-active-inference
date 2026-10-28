"""Fixed-budget label-free training and registered hierarchy go/no-go plan."""
import argparse,json,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from pai.thinker.predictive import PredictiveConfig,HierarchicalPredictor,FlatPredictor,parameters,predictive_loss
from pai.thinker.hier_data import load_split,normalization,batch_windows
from thinker_hier_collect import save,digest

KINDS=['hierarchical','flat','amortized']
SEEDS=[93000,93001,93002]

def build(kind,cfg):
    return FlatPredictor(cfg,75) if kind=='flat' else HierarchicalPredictor(cfg,iterative=kind=='hierarchical')

def source_hashes():
    return {str(p).replace('\\','/'):digest(p) for pattern in ('pai/thinker/predictive.py','pai/thinker/hier_*.py','scripts/thinker_hier_*.py') for p in Path('.').glob(pattern)}

def register():
    path=Path('results/thinker_hier_plan.json')
    if path.exists():return json.loads(path.read_text(encoding='utf8'))
    cfg=PredictiveConfig()
    plan=dict(version=1,models=KINDS,training_seeds=SEEDS,model=asdict(cfg),flat_width=75,
       parameters={k:parameters(build(k,cfg)) for k in KINDS},steps=1200,batch_size=16,sequence_length=24,
       lr=.001,selection='fixed final optimizer step, no label-based model selection',
       data_seeds=dict(train=list(range(90000,90200)),development=list(range(91000,91040)),final=list(range(92000,92040))),
       counts_per_condition=dict(train=200,development=40,final=40),
       go_rule='GO only if hierarchical minus flat mean cluster accuracy >= 0.05 AND paired training-seed Student-t 95% lower bound > 0 AND hierarchical mean overall prediction MSE < flat; otherwise NO-GO',
       mapping='train-only standardization/PCA16/KMeans5; Hungarian and fixed-C logistic probe use development labels only',
       representation='mean and std over all causal state steps of each episode; no onset or true label selects a state',
       signatures='autocorrelation time in steps; event-aligned error; inner iterations vs own surprise and condition',
       optional_local_updates=False)
    assert max(plan['parameters'].values())/min(plan['parameters'].values())<1.1
    save(path,plan);return plan

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--register',action='store_true');ap.add_argument('--kind',choices=KINDS);ap.add_argument('--seed',type=int);ap.add_argument('--device',default='cuda')
    args=ap.parse_args();plan=register()
    if args.register:print(json.dumps(plan,indent=2));raise SystemExit
    if args.kind is None or args.seed not in SEEDS:raise ValueError('Use registered model and seed')
    rows,streams=load_split('train');assert len(streams)==1000
    cfg=PredictiveConfig(**plan['model']);torch.set_num_threads(1);torch.manual_seed(args.seed)
    model=build(args.kind,cfg).to(args.device);norm=normalization(streams)
    optimizer=torch.optim.AdamW(model.parameters(),lr=plan['lr'],weight_decay=.0001)
    rng=np.random.default_rng(args.seed);start=time.time();history=[]
    prefix=f'results/thinker_hier_{args.kind}_s{args.seed}'
    output=Path(prefix+'.pt')
    if output.exists():raise FileExistsError(output)
    sources=source_hashes()
    for step in range(plan['steps']):
        s,a,m,y=batch_windows(streams,norm,rng,plan['batch_size'],plan['sequence_length'],args.device)
        model.train();optimizer.zero_grad(set_to_none=True)
        prediction,traces=model(s,a,m)
        loss=predictive_loss(prediction,y,model)
        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss; retain failed variant')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if step%100==0 or step+1==plan['steps']:
            history.append(dict(step=step+1,nll=float(loss.detach()),mse=float((prediction-y).square().mean().detach()),
                   iterations=float(traces['iterations'].float().mean()),elapsed_s=time.time()-start))
            print(args.kind,args.seed,history[-1],flush=True)
    payload=dict(kind=args.kind,seed=args.seed,model=model.state_dict(),config=asdict(cfg),normalization=norm,
                 plan_sha256=digest('results/thinker_hier_plan.json'),source_hashes=sources)
    torch.save(payload,output)
    save(prefix+'_training.json',dict(kind=args.kind,seed=args.seed,parameters=parameters(model),steps=plan['steps'],
         train_wall_s=time.time()-start,history=history,checkpoint=str(output).replace('\\','/'),checkpoint_sha256=digest(output),
         source_hashes=sources,streams={r['path']:r['sha256'] for r in rows},command=['python',*__import__('sys').argv]))
