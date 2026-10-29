"""PPO/GRU-PPO training with registered budget; no demonstration pretraining."""
import argparse
from collections import deque
from pathlib import Path
import time

import numpy as np
import torch

from pai.baselines.ppo import ActorCritic,gae,loss
from pai.baselines.vector import Vector
from pai.baselines.checkpoint import random_state,restore_random,atomic_save
from arm_baseline_plan import register,PREFIX,save,sources,digest


def train(kind,seed):
    plan=register();torch.set_num_threads(1);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    if seed not in plan['training_seeds']:raise ValueError('Unregistered seed')
    recurrent=kind=='recurrent_ppo';policy=ActorCritic(recurrent=recurrent).cuda()
    optimizer=torch.optim.Adam(policy.parameters(),lr=plan['lr'],eps=1e-5)
    vec=Vector(plan['n_envs'],plan['training_seeds'].index(seed),kind+str(seed));obs=torch.tensor(vec.reset(),device='cuda')
    hidden=torch.zeros(plan['n_envs'],128,device='cuda');reset=torch.ones(plan['n_envs'],device='cuda')
    updates=plan['budget_steps']//(plan['rollout']*plan['n_envs']);start=time.time();history=[];episodes=[];recent=deque(maxlen=100)
    prefix=PREFIX+f'{kind}_s{seed}';fingerprints=sources()
    if Path(prefix+'.pt').exists() and (Path(prefix+'_training.json').exists() or not Path(prefix+'_resume.pt').exists()):
        raise FileExistsError('Completed model already exists')
    stop='budget';best_windows=[];first_update=0;elapsed_before=0.;finished=False
    resume_path=prefix+'_resume.pt'
    if Path(resume_path).exists():
        state=torch.load(resume_path,map_location='cpu',weights_only=False)
        if state['source_hashes']!=fingerprints or state['plan_sha256']!=digest(PREFIX+'plan.json'):
            raise ValueError('Resume configuration/source mismatch')
        policy.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        restored=vec.restore(state['workers'])
        if not np.array_equal(restored,state['obs'].numpy()):raise ValueError('Resume observation mismatch')
        obs=state['obs'].cuda();hidden=state['hidden'].cuda();reset=state['reset'].cuda()
        history=state['history'];episodes=state['episodes'];recent=deque(state['recent'],maxlen=100)
        best_windows=state['best_windows'];first_update=state['next_update'];elapsed_before=state['wall_s']
        stop=state['stop'];finished=state['finished'];rng.bit_generator.state=state['shuffle_rng']
        restore_random(state['random']);print('Resumed',kind,seed,first_update*1024,flush=True)
    def wall():return elapsed_before+time.time()-start
    def checkpoint(next_update,finished):
        atomic_save(resume_path,dict(model=policy.state_dict(),optimizer=optimizer.state_dict(),
            source_hashes=fingerprints,plan_sha256=digest(PREFIX+'plan.json'),workers=vec.snapshot(),
            obs=obs.cpu(),hidden=hidden.cpu(),reset=reset.cpu(),history=history,episodes=episodes,
            recent=list(recent),best_windows=best_windows,next_update=next_update,wall_s=wall(),
            stop=stop,finished=finished,shuffle_rng=rng.bit_generator.state,random=random_state()))
    try:
        steps=first_update*plan['rollout']*plan['n_envs']
        for update in range(first_update,updates) if not finished else []:
            buffer={k:[] for k in ('obs','hidden','reset','raw','logprob','value','reward','next_value','done')}
            for t in range(plan['rollout']):
                with torch.no_grad():a,lp,v,next_hidden,raw=policy.act(obs,hidden,reset)
                (terminal,reward,done,truncated,following),summaries=vec.step(a.cpu().numpy())
                with torch.no_grad():_,next_value,_=policy(torch.tensor(terminal,device='cuda'),next_hidden)
                next_value=next_value*torch.tensor(np.logical_or(~done.astype(bool),truncated.astype(bool)),device='cuda')
                data=dict(obs=obs,hidden=hidden,reset=reset,raw=raw,logprob=lp,value=v,
                    reward=torch.tensor(reward,device='cuda',dtype=torch.float32),next_value=next_value,
                    done=torch.tensor(done,device='cuda',dtype=torch.float32))
                for k in buffer:buffer[k].append(data[k].detach())
                for summary in summaries:
                    if summary is not None:episodes.append(summary);recent.append(summary['return_'])
                obs=torch.tensor(following,device='cuda');hidden=next_hidden.detach() if recurrent else hidden
                reset=data['done']
            b={k:torch.stack(v) for k,v in buffer.items()}
            adv,returns=gae(b['reward'],b['value'],b['next_value'],b['done'])
            adv=(adv-adv.mean())/(adv.std()+1e-8)
            steps=(update+1)*plan['rollout']*plan['n_envs']
            optimizer.param_groups[0]['lr']=plan['lr']*(1-update/updates)
            exhausted=False;stats={}
            for epoch in range(plan['epochs']):
                if recurrent:
                    chunks=[(t,e) for t in range(0,plan['rollout'],plan['sequence']) for e in range(plan['n_envs'])]
                    rng.shuffle(chunks);batches=[chunks[i:i+8] for i in range(0,len(chunks),8)]
                else:batches=np.array_split(rng.permutation(plan['rollout']*plan['n_envs']),4)
                for batch in batches:
                    if recurrent:
                        def take(x):return torch.stack([x[t:t+plan['sequence'],e] for t,e in batch],1)
                        initial=torch.stack([b['hidden'][t,e] for t,e in batch])
                        objective,stats=loss(policy,take(b['obs']),take(b['raw']),take(b['logprob']),take(adv),take(returns),initial,take(b['reset']))
                    else:
                        ids=torch.tensor(batch,device='cuda')
                        def take(x):return x.flatten(0,1)[ids]
                        objective,stats=loss(policy,take(b['obs']),take(b['raw']),take(b['logprob']),take(adv),take(returns))
                    if not torch.isfinite(objective):raise ValueError('Nonfinite objective; retain failed run')
                    optimizer.zero_grad(set_to_none=True);objective.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),.5);optimizer.step()
                    if stats['approx_kl']>.03:exhausted=True;break
                if exhausted:break
            if (update+1)%25==0 or update==updates-1:
                row=dict(steps=steps,episodes=len(episodes),rolling_return=float(np.mean(recent)),
                    rolling_success=float(np.mean([x['success'] for x in episodes[-100:]])),wall_s=wall(),**stats)
                history.append(row);best_windows.append(row['rolling_return']);print(kind,seed,row,flush=True)
                save(prefix+'_progress.json',dict(history=history,last_step=steps))
                if steps>=750592 and len(best_windows)>=8:
                    previous=max(best_windows[-8:-4]);current=max(best_windows[-4:])
                    if current<=previous+.01*max(1.,abs(previous)):stop='registered plateau';finished=True
                finished=finished or update==updates-1
                checkpoint(update+1,finished)
                if finished:break
        payload=dict(model=policy.state_dict(),recurrent=recurrent,seed=seed,kind=kind,
            plan_sha256=digest(PREFIX+'plan.json'),source_hashes=fingerprints)
        atomic_save(prefix+'.pt',payload)
        save(prefix+'_training.json',dict(kind=kind,seed=seed,environment_decisions=steps,
            controlled_50Hz_steps=steps*5,passive_settle_steps=len(episodes)*25,
            reset_settle_steps=(len(episodes)+plan['n_envs'])*plan['config']['env']['settle_steps'],
            wall_s=wall(),stop=stop,history=history,episodes=episodes,
            parameters=sum(p.numel() for p in policy.parameters()),source_hashes=fingerprints,
            checkpoint_sha256=digest(prefix+'.pt')))
    finally:vec.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['ppo','recurrent_ppo'],required=True);ap.add_argument('--seed',type=int,required=True)
    a=ap.parse_args();train(a.kind,a.seed)
