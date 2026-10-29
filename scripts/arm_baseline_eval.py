"""Frozen paired episodes, shared disturbances and score-only event ledgers."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from pai.baselines.arm import ArmTask,CONDITIONS
from pai.baselines.ppo import ActorCritic
from pai.causes.inference import Calibration,calibrate
from pai.agents.slice_agent import run_episode
from pai.config import load_config
from pai.envs import TabletopEnv
from pai.memory.report import episode_report
from pai.world.model import load_world_model
from slice_eval import disturbance_for
from arm_baseline_plan import PREFIX,register,save,digest,sources


def calibration():
    plan=register();cfg=load_config('configs/tabletop.yaml',['env.render_images=false'])
    env=TabletopEnv(cfg.env,disturbances=[]);wm=load_world_model(cfg.slice.world_model,'cuda');raw=[]
    try:
        for seed in plan['calibration_seeds']:run_episode(env,wm,cfg,-1,seed,device='cuda',adapt=False,raw=raw)
    finally:env.close()
    c=calibrate(np.stack([r[0] for r in raw]),np.stack([r[1] for r in raw]),[r[2] for r in raw])
    save(PREFIX+'calibration.json',dict(floor=c.floor.tolist(),threshold=c.threshold,
        step_floors={k:v.tolist() for k,v in c.step_floors.items()},step_thresholds=c.step_thresholds,calibration_decisions=len(raw)))


def freeze():
    plan=register();files=[PREFIX+'plan.json',PREFIX+'calibration.json',PREFIX+'tests_final.json',plan['config']['slice']['world_model']]
    for kind in ('ppo','recurrent_ppo'):
        for seed in plan['training_seeds']:files += [PREFIX+f'{kind}_s{seed}'+x for x in ('.pt','_training.json')]
    if json.loads(Path(PREFIX+'tests_final.json').read_text())['exit_code']:raise ValueError('Tests must pass')
    payload=dict(files={p:digest(p) for p in files},sources={**sources(),**plan['agent_sources'],'scripts/slice_eval.py':digest('scripts/slice_eval.py')},rule='No final episodes seen before freezing')
    path=Path(PREFIX+'freeze.json')
    if path.exists():raise FileExistsError('Already frozen')
    save(path,payload)


def evaluate(method,seed,split='final'):
    plan=register();frozen=json.loads(Path(PREFIX+'freeze.json').read_text())
    for group in ('files','sources'):
        for path,sha in frozen[group].items():
            if digest(path)!=sha:raise ValueError('Frozen artifact changed: '+path)
    n=plan[split+'_n'];base=plan[split+'_seed_base'];records=[]
    output=PREFIX+f'{split}_{method}'+(f'_s{seed}' if seed is not None else '')+'.json'
    if Path(output).exists():
        old=json.loads(Path(output).read_text());records=old['records']
    if method in ('ppo','recurrent_ppo'):
        payload=torch.load(PREFIX+f'{method}_s{seed}.pt',map_location='cuda',weights_only=False)
        policy=ActorCritic(recurrent=payload['recurrent']).cuda();policy.load_state_dict(payload['model']);policy.eval()
        task=ArmTask()
    else:
        cfg=load_config('configs/tabletop.yaml',['env.render_images=false']);env=TabletopEnv(cfg.env,disturbances=[])
        wm=load_world_model(cfg.slice.world_model,'cuda');v=json.loads(Path(PREFIX+'calibration.json').read_text())
        c=Calibration(floor=np.array(v['floor']),threshold=v['threshold'],step_floors={k:np.array(x) for k,x in v['step_floors'].items()},step_thresholds=v['step_thresholds'])
    try:
        for i in range(n):
            if any(r['index']==i for r in records):continue
            condition=CONDITIONS[i//(n//5)];layout=base+i
            disturbances=disturbance_for(condition,np.random.default_rng(plan['disturbance_rng_bases'][split]+i))
            row=dict(index=i,seed=layout,condition=condition,disturbance_draw_sha256=__import__('hashlib').sha256(json.dumps(disturbances,sort_keys=True).encode()).hexdigest())
            if method in ('ppo','recurrent_ppo'):
                obs=task.reset(layout,disturbances);hidden=torch.zeros(1,128,device='cuda');done=False
                while not done:
                    with torch.no_grad():a,_,_,hidden,_=policy.act(torch.tensor(obs[None],device='cuda'),hidden,deterministic=True)
                    obs,_,done,_=task.step(a[0].cpu().numpy())
                score=task.score();events=score.pop('events')
                row.update(score,explanation=None,inferred=None)
            else:
                record=run_episode(env,wm,cfg,i,layout,disturbances,device='cuda',calibration=c,adapt=method=='full_agent')
                # Never give the report generator an answer key. Keep only the own
                # cause/surprise/adaptation lines, not its scorer-success header.
                report=episode_report(replace(record,true_disturbances=[],events=[]))
                row.update(success=record.success,steps=record.steps,inferred=record.inferred_cause,
                    posterior=record.cause_posterior,adaptations=record.adaptations,
                    explanation='\n'.join(line for line in report.splitlines() if line.strip().startswith(('My explanation:','What I changed:','Surprise:'))))
                events=record.events
            row['realized_disturbances']=[e for e in events if e['type']=='disturbance_start']
            records.append(row);save(output,dict(method=method,training_seed=seed,split=split,records=records,complete=len(records)==n))
            print(method,seed,i,condition,row['success'],flush=True)
    finally:
        if method in ('ppo','recurrent_ppo'):task.close()
        else:env.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['calibration','freeze','evaluate']);ap.add_argument('--method',choices=['ppo','recurrent_ppo','full_agent','no_adapt']);ap.add_argument('--seed',type=int);ap.add_argument('--split',default='final',choices=['final','development'])
    a=ap.parse_args();torch.set_num_threads(1)
    if a.stage=='calibration':calibration()
    elif a.stage=='freeze':freeze()
    else:evaluate(a.method,a.seed,a.split)
