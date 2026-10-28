"""Collect RGB/proprioceptive streams; answer keys and reference evidence are separate."""
import argparse,hashlib,json,re,time
from pathlib import Path
import numpy as np
import torch
from pai.agents import slice_agent
from pai.causes.inference import SurpriseMonitor
from pai.config import load_config
from pai.envs import TabletopEnv
from pai.world.model import load_world_model
from pai.thinker.hier_sensors import OwnPixelPerception,sensor_vector
from slice_eval import CONDITIONS,disturbance_for


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,data):
    text=json.dumps(data,indent=2,allow_nan=False)
    if re.search(r'[A-Za-z]:[\\/]|/Users/|/home/',text):raise ValueError('Absolute path in results')
    p=Path(p);tmp=Path(str(p)+'.tmp');tmp.write_text(text,encoding='utf8');tmp.replace(p)

class PassiveMonitor(SurpriseMonitor):
    def add(self,ev,raw=None):
        self.history.append(ev);self.raw_history.append(raw if raw is not None else ev)
        return None
    def episode_verdict(self):return None

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split',choices=['pilot','train','development','final'],required=True)
    ap.add_argument('--worker',type=int,default=0);ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--count',type=int,default=None);ap.add_argument('--device',default='cuda')
    args=ap.parse_args();torch.set_num_threads(1)
    count=args.count or dict(pilot=1,train=200,development=40,final=40)[args.split]
    base=dict(pilot=89800,train=90000,development=91000,final=92000)[args.split]
    # These budgets are data-generation settings, equal for every condition/split.
    cfg=load_config('configs/tabletop.yaml',['env.render_images=false','env.image_size=160',
        'slice.samples=16','slice.iterations=1','slice.horizon=6','slice.max_plan_steps=80',
        'slice.watch_settle=false','slice.settle_steps=5'])
    wm=load_world_model(cfg.slice.world_model,args.device);env=TabletopEnv(cfg.env,disturbances=[])
    perception=OwnPixelPerception(env)
    slice_agent.SurpriseMonitor=PassiveMonitor
    prefix=f'results/thinker_hier_{args.split}'
    manifest=Path(prefix+f'_manifest{args.worker}.json');truth_path=Path(prefix+f'_truth{args.worker}.json')
    hashes={str(p).replace('\\','/'):digest(p) for pattern in ('pai/thinker/hier_sensors.py','scripts/thinker_hier_collect.py','pai/agents/slice_agent.py','configs/tabletop.yaml') for p in Path('.').glob(pattern)}
    meta=dict(split=args.split,base_seed=base,count_per_condition=count,worker=args.worker,workers=args.workers,
              source_hashes=hashes,world_model_sha256=digest(cfg.slice.world_model),config=cfg.to_dict(),
              regime='RGB with given calibrated color/shape geometry; proprioception; adapt=False',
              command=['python',*__import__('sys').argv],episodes=[],complete=False)
    truth=dict(split=args.split,episodes=[])
    if manifest.exists():
        old=json.loads(manifest.read_text(encoding='utf8'))
        for k in ('count_per_condition','source_hashes','config','workers','world_model_sha256'):
            if old[k]!=meta[k]:raise ValueError('Cannot resume changed collection: '+k)
        meta=old;truth=json.loads(truth_path.read_text(encoding='utf8'))
    t0=time.time()
    for ci,condition in enumerate(CONDITIONS):
        for i in range(count):
            index=ci*count+i
            if index%args.workers!=args.worker:continue
            if any(r['index']==index for r in meta['episodes']):continue
            seed=base+i
            disturbances=disturbance_for(condition,np.random.default_rng(base+10000+ci*1000+i))
            raw=[];ev=[];senses=[]
            record=slice_agent.run_episode(env,wm,cfg,index,seed,disturbances,device=args.device,
                    perceive=perception,adapt=False,raw=raw,evidence=ev,sensory=senses)
            if not senses:raise ValueError('Empty sensory episode')
            state=np.stack([sensor_vector(senses[0]['before'],perception.features[0])]+
                           [sensor_vector(s['after'],perception.features[j+1]) for j,s in enumerate(senses)])
            action=np.stack([s['action'] for s in senses]);holding=np.array([s['holding'] for s in senses])
            phase=np.array([s['phase'] for s in senses]);context=np.c_[holding,np.eye(9)[np.minimum(phase,8)]]
            path=Path(prefix+f'_{index:04d}.npz')
            np.savez_compressed(path,state=state,action=action,context=context,
                tokens=np.stack([senses[0]['before']]+[s['after'] for s in senses]),
                q=np.stack([s['q'] for s in senses]),dq=np.stack([s['dq'] for s in senses]))
            ref=Path(prefix+f'_reference_{index:04d}.npz')
            np.savez_compressed(ref,residual=np.stack([r[0] for r in raw]),sigma=np.stack([r[1] for r in raw]),
                step=np.array([r[2] for r in raw]),holding=holding,phase=phase,action=action)
            truth['episodes'].append(dict(index=index,seed=seed,assigned_condition=condition,
                disturbances=record.true_disturbances,events=record.events,success=record.success,
                reference=str(ref).replace('\\','/')))
            meta['episodes'].append(dict(index=index,path=str(path).replace('\\','/'),sha256=digest(path),steps=len(senses)))
            save(truth_path,truth);save(manifest,meta)
            print(args.split,index,condition,seed,len(senses),record.success,'elapsed',round(time.time()-t0),flush=True)
    meta['complete']=True;save(manifest,meta)
