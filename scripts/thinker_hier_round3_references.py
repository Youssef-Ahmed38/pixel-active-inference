"""Label/teacher-based references, kept separate from predictive training."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from pai.causes.inference import Calibration,SurpriseMonitor,StepEvidence,calibrate,combined_sigma
from pai.thinker.hrm import HRM
from pai.thinker.hier_analysis import cause,rate
from pai.thinker.hier3_tokens import truth_rows,load_split
from thinker_hier_collect import digest
from thinker_hier_round2 import save
from pai.thinker.hier_data import normalization,normalized

CALIBRATION='results/thinker_hier3_reference_calibration.json'

def final(device):
    value=json.loads(Path(CALIBRATION).read_text(encoding='utf8'))
    c=Calibration(floor=np.array(value['floor']),threshold=value['threshold'],step_floors={k:np.array(v) for k,v in value['step_floors'].items()},step_thresholds=value['step_thresholds'])
    models=[];hashes={CALIBRATION:digest(CALIBRATION)}
    for seed in (0,1,2):
        path=f'results/thinker/checkpoints/real_c1000_hrm_s{seed}.pt';p=torch.load(path,map_location=device,weights_only=False)
        model=HRM(p['in_dim'],2,20,{'cause':len(p['causes'])},**p['config']['model']['hrm']).to(device)
        model.load_state_dict(p['model']);model.eval();models.append((model,p['causes']));hashes[path]=digest(path)
    stream_rows,streams=load_split('final')
    _,training=load_split('train');norm=normalization(training)
    stream_by_id={r['index']:normalized(s,norm) for r,s in zip(stream_rows,streams)}
    records=[]
    for index,row in sorted(truth_rows('final').items()):
        d=np.load(row['reference'],allow_pickle=False);monitor=SurpriseMonitor(.1,calibration=c);zs=[]
        for t,(res,sig,name,act,held) in enumerate(zip(d['residual'],d['sigma'],d['step'],d['action'],d['holding'])):
            sigma=combined_sigma(sig,c,str(name));ev=StepEvidence(t,res,sigma,act,bool(held),str(name));monitor.add(ev,ev);zs.append(res/sigma)
        verdict=monitor.episode_verdict();bayes=verdict.best if verdict else 'none'
        z=np.stack(zs).astype(np.float32);peak=int((z*z).sum(1).argmax());lo=max(0,peak-15)
        x=np.c_[np.arcsinh(z),d['holding'],np.eye(10)[np.minimum(d['phase'],9)]].astype(np.float32)
        mask=np.zeros(20,np.float32);mask[:len(x[lo:lo+20])]=1
        padded=np.zeros((20,x.shape[1]),np.float32);padded[:int(mask.sum())]=x[lo:lo+20]
        ctx=np.array([[np.log1p(.5*np.square(z[peak]).sum()),mask.mean()]],np.float32)
        students=[];segments=[]
        for model,names in models:
            with torch.no_grad():out=model.think(torch.tensor(padded[None],device=device),torch.tensor(mask[None],device=device),torch.tensor(ctx,device=device))
            students.append(names[int(out.logits['cause'].argmax(-1))]);segments.append(int(out.segments[0]))
        stream=stream_by_id[index];err=np.square(stream['state'][1:]-stream['state'][:-1]).mean(1)
        onset=min((max(0,(int(e['t'])-1)//5) for e in row['disturbances']),default=None)
        post=err[onset:onset+20] if onset is not None else np.array([])
        records.append(dict(index=index,seed=row['seed'],label=cause(row),bayes=bayes,hrm=students,hrm_segments=segments,
             persistence_mse=float(err.mean()),persistence_post20=float(post.mean()) if len(post)==20 else None,persistence_post_curve=post.tolist()))
    save('results/thinker_hier3_references_final.json',dict(records=records,hashes=hashes,
          bayes=rate(sum(r['label']==r['bayes'] for r in records),len(records)),
          hrm=[rate(sum(r['label']==r['hrm'][j] for r in records),len(records)) for j in range(3)],
          persistence_mse=float(np.mean([r['persistence_mse'] for r in records])),
          persistence_post20=float(np.mean([r['persistence_post20'] for r in records if r['persistence_post20'] is not None])),
          note='References use given likelihood forms or supervised teacher labels; not unsupervised competitors. Privileged default token regime; references are supervised/given-likelihood and not matched competitors.'))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['final']);ap.add_argument('--device',default='cuda');args=ap.parse_args();torch.set_num_threads(1)
    final(args.device)
