"""Fit a separate recurrent chess residual and supervised benefit-based halting."""
import argparse,copy,gzip,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from pai.chess.adaptive_hrm import AdaptiveHRMValue,stopping_targets
from pai.chess.hrm_value import board_tensors,mirror_observation
from pai.chess.distill import observation_from_fen
from pai.chess.evaluation import LearnedPositionValue
from pai.chess.hrm_training import orbit_key
from pai.thinker.hrm import HRM


def read(path):
    with gzip.open(path,'rt',encoding='utf-8') as f:return [json.loads(line) for line in f]


def save(path,data):
    blob=json.dumps(data,allow_nan=False,separators=(',',':')).encode()
    if len(blob)>5_000_000 and not str(path).endswith('.gz'):raise ValueError('gzip required')
    Path(path).write_bytes(gzip.compress(blob,mtime=0) if str(path).endswith('.gz') else blob)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensors(rows,baseline):
    obs=[observation_from_fen(r[0]) for r in rows]
    y=torch.tensor([r[1]*(1 if o.turn else -1) for r,o in zip(rows,obs)],dtype=torch.float32)
    base=torch.tensor(baseline.predict_many(obs),dtype=torch.float32)
    x,mask,ctx=board_tensors(obs+[mirror_observation(o) for o in obs])
    return obs,y,base,x,mask,ctx


@torch.no_grad()
def metrics(value,rows):
    obs,y,base,*_=tensors(rows,value.baseline)
    value.reset_diagnostics();pred=value.predict_many(obs)
    counts=value.halting_stats()
    return {'positions':len(obs),'mae_cp':float(np.mean(np.abs(pred-y.numpy()))),
            'baseline_mae_cp':float(np.mean(np.abs(base.numpy()-y.numpy()))),'halting':counts}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--limit',type=int,default=4000);p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--halt-epochs',type=int,default=25);p.add_argument('--seed',type=int,default=91400)
    p.add_argument('--cost-cp',type=float,default=2.);p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--output',default='results/chess_hrm_adaptive_model.pt')
    p.add_argument('--report',default='results/chess_hrm_training.json.gz')
    args=p.parse_args()
    if Path(args.output).exists() or Path(args.report).exists():raise FileExistsError('retain previous training outputs')
    torch.set_num_threads(1);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    baseline=LearnedPositionValue.load('runs/chess_strength_data/linear_candidate.json')
    raw=read('runs/chess_strength_data/train.jsonl.gz')
    selected=[raw[i] for i in rng.choice(len(raw),min(args.limit,len(raw)),replace=False)]
    seen=set();train=[]
    for row in selected:
        key=orbit_key(observation_from_fen(row[0]))
        if key not in seen:train.append(row);seen.add(key)
    val=[r for r in read('runs/chess_strength_data/val.jsonl.gz')
         if orbit_key(observation_from_fen(r[0])) not in seen][:800]
    config=dict(in_dim=16,ctx_dim=5,max_len=64,heads={'value':1},d_model=32,n_heads=4,
                h_layers=1,l_layers=1,h_cycles=1,l_steps=1,max_segments=6,patch=8)
    model=HRM(**config)
    torch.nn.init.zeros_(model.readout.heads['value'].weight);torch.nn.init.zeros_(model.readout.heads['value'].bias)
    value=AdaptiveHRMValue(model,baseline,config)
    obs,y,base,x,mask,ctx=tensors(train,baseline);n=len(obs)
    opt=torch.optim.AdamW(model.parameters(),lr=.0008,weight_decay=.01)
    history=[];best=float('inf');state=None
    for epoch in range(args.epochs):
        model.train()
        for start in range(0,n,args.batch_size):
            # Shuffle once per epoch, preserving mirrored supervision.
            if start==0:order=torch.randperm(n)
            ix=order[start:start+args.batch_size];paired=torch.cat((ix,ix+n))
            outs=model.forward_segments(x[paired],mask[paired],ctx[paired])
            losses=[]
            for out in outs:
                raw=out.logits['value'].squeeze(-1);pred=(raw[:len(ix)]-raw[len(ix):])/2
                losses.append(F.smooth_l1_loss(pred,(y[ix]-base[ix])/500))
            loss=torch.stack(losses).mean();opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        model.eval()
        # Choose the backbone on the validation full-depth readout, never final games.
        vobs,vy,*_=tensors(val,baseline);pred=value.predict_many(vobs,segments=6)
        mae=float(np.mean(np.abs(pred-vy.numpy())))
        history.append({'epoch':epoch+1,'full_depth_validation_mae_cp':mae})
        if mae<best:best=mae;state=copy.deepcopy(model.state_dict())
        print('value epoch',epoch+1,'validation MAE',round(mae,2),flush=True)
    model.load_state_dict(state);model.eval()
    states=[];targets=[];oracle_steps=[]
    with torch.no_grad():
        for start in range(0,n,args.batch_size):
            ix=torch.arange(start,min(start+args.batch_size,n));paired=torch.cat((ix,ix+n))
            xt=model.encoder(x[paired],mask[paired],ctx[paired]);h,l=model._init_state(xt)
            hs=[];errors=[]
            for _ in range(6):
                h,l=model._segment(h,l,xt);hs.append(h[:,0].clone())
                raw=model._read(h).logits['value'].squeeze(-1)
                errors.append(torch.abs(base[ix]+500*(raw[:len(ix)]-raw[len(ix):])/2-y[ix]))
            labels=stopping_targets(torch.stack(errors),args.cost_cp)
            first=torch.full((len(ix),),6,dtype=torch.long)
            for m in range(5,-1,-1):first[labels[m]==0]=m+1
            oracle_steps.extend(first.tolist())
            states.append(torch.stack(hs).reshape(-1,32))
            targets.append(torch.cat((labels,labels),dim=1).reshape(-1))
    hs=torch.cat(states);labels=torch.cat(targets)
    for parameter in model.parameters():parameter.requires_grad_(False)
    for parameter in model.q_head.parameters():parameter.requires_grad_(True)
    torch.nn.init.zeros_(model.q_head.weight);torch.nn.init.zeros_(model.q_head.bias)
    opt=torch.optim.Adam(model.q_head.parameters(),lr=.005)
    counts=torch.bincount(labels,minlength=2).float();weight=counts.sum()/(2*counts.clamp_min(1))
    halt_history=[];best=float('inf');best_q=None
    for epoch in range(args.halt_epochs):
        order=torch.randperm(len(hs))
        for start in range(0,len(hs),256):
            ix=order[start:start+256];loss=F.cross_entropy(model.q_head(hs[ix]),labels[ix],weight=weight)
            opt.zero_grad();loss.backward();opt.step()
        metric=metrics(value,val);objective=metric['mae_cp']+args.cost_cp*metric['halting']['mean_segments']
        halt_history.append({'epoch':epoch+1,**metric,'selection_objective':objective})
        if objective<best:best=objective;best_q=copy.deepcopy(model.q_head.state_dict())
        print('halt epoch',epoch+1,'validation',round(metric['mae_cp'],2),'steps',round(metric['halting']['mean_segments'],2),flush=True)
    model.q_head.load_state_dict(best_q)
    value.summary={'learned_halting':True,'seed':args.seed,'training_positions':sorted(seen),
                   'training_examples':n,'validation_examples':len(val),'cost_cp_per_segment':args.cost_cp,
                   'target':'future teacher-error reduction exceeds additional recurrence cost',
                   'given':'architecture, six-segment cap, two-cp compute cost, board tokens, teacher labels',
                   'baseline':'runs/chess_strength_data/linear_candidate.json'}
    value.save(args.output)
    report={'seed':args.seed,'model':args.output,'model_sha256':sha(args.output),'value_history':history,
            'halt_history':halt_history,'validation':metrics(value,val),'training_oracle_steps':oracle_steps,
            'training_label_counts':counts.tolist(),'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in
              [ROOT/'pai/chess/adaptive_hrm.py',ROOT/'pai/thinker/hrm.py',Path(__file__)]},
            'inputs':{p:sha(p) for p in ['runs/chess_strength_data/train.jsonl.gz','runs/chess_strength_data/val.jsonl.gz']}}
    save(args.report,report)


if __name__=='__main__':main()
