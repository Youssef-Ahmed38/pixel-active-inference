"""Timing/interface probe only: random untrained policy, no weight updates."""
import time
import torch
from pai.baselines.ppo import ActorCritic
from pai.baselines.vector import Vector
from arm_baseline_plan import PREFIX,save

if __name__=='__main__':
    torch.set_num_threads(1);torch.manual_seed(123456)
    vec=Vector(4,9);model=ActorCritic().cuda();start=time.time()
    try:
        obs=torch.tensor(vec.reset(),device='cuda')
        initialized=time.time()
        for i in range(256):
            with torch.no_grad():action,_,_,_,_=model.act(obs)
            values,_=vec.step(action.cpu().numpy());obs=torch.tensor(values[4],device='cuda')
        seconds=time.time()-start
        save(PREFIX+'timing_probe.json',dict(decisions=1024,wall_s=seconds,decisions_per_s=1024/seconds,initialization_s=initialized-start,simulation_s=time.time()-initialized,learning_updates=0,layout_range='19000000+; separate from registered training/dev/final',purpose='Interface/timing probe, not a trained baseline result'))
        print('Probe:',1024,'decisions in',seconds,'seconds; initialization',initialized-start,'simulation',time.time()-initialized,flush=True)
    finally:vec.close()
