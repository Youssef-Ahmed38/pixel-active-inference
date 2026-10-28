"""Opt-in chess HRM residual with a teacher-benefit-trained stopping head.

Inference consumes board tokens only. The teacher is confined to the separate
training script; existing HRM users and fixed-depth checkpoints are unchanged.
"""
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from pai.chess.evaluation import LearnedPositionValue
from pai.chess.hrm_value import board_tensors, mirror_observation
from pai.thinker.hrm import HRM


def stopping_targets(errors, cost_cp=2.):
    """Stop if no future readout improves teacher error enough to pay its cost.

    errors: [segments, examples], detached before supervision. Ties stop.
    Class 0=halt, 1=continue. The cap is given; earlier decisions are learned.
    """
    errors=errors.detach()
    target=torch.zeros_like(errors,dtype=torch.long)
    for m in range(len(errors)-1):
        costs=torch.arange(1,len(errors)-m,device=errors.device)[:,None]*cost_cp
        future=(errors[m+1:]+costs).amin(0)
        target[m]=(future<errors[m]).long()
    return target


class AdaptiveHRMValue:
    def __init__(self,model,baseline,config,summary=None,residual_scale=500.):
        self.model=model.cpu().eval();self.baseline=baseline;self.config=config
        self.summary=summary or {};self.residual_scale=residual_scale
        self.reset_diagnostics()

    def reset_diagnostics(self):
        self._segments=Counter();self._boards=0

    def halting_stats(self):
        n=sum(self._segments.values())
        return {'boards':self._boards,'views':n,'segments_histogram':dict(self._segments),
                'mean_segments':sum(k*v for k,v in self._segments.items())/n if n else None,
                'low_level_steps_per_segment':self.model.steps_per_segment}

    @torch.no_grad()
    def predict_many(self,observations,segments=None):
        obs=list(observations)
        if not obs:return np.empty(0)
        if segments is not None and segments<1:raise ValueError('positive segment cap required')
        values=[]
        for start in range(0,len(obs),32):
            chunk=obs[start:start+32];paired=chunk+[mirror_observation(o) for o in chunk]
            x,mask,ctx=board_tensors(paired)
            result=self.model.think(x,mask,ctx,max_segments=segments,halt=segments is None)
            raw=result.logits['value'].squeeze(-1).numpy()
            residual=(raw[:len(chunk)]-raw[len(chunk):])/2
            values.extend(self.baseline.predict_many(chunk)+self.residual_scale*residual)
            self._segments.update(map(int,result.segments.tolist()));self._boards+=len(chunk)
        return np.asarray(values)

    def predict(self,observation):return float(self.predict_many([observation])[0])

    def save(self,path):
        torch.save({'schema':1,'type':'chess_adaptive_hrm_value','config':self.config,
                    'state_dict':self.model.state_dict(),'baseline_weights':self.baseline.weights.tolist(),
                    'baseline_summary':self.baseline.summary,'summary':self.summary,
                    'residual_scale':self.residual_scale},Path(path))

    @classmethod
    def load(cls,path):
        d=torch.load(path,map_location='cpu',weights_only=True)
        if d.get('type')!='chess_adaptive_hrm_value' or d.get('schema')!=1:raise ValueError('unsupported adaptive HRM')
        m=HRM(**d['config']);m.load_state_dict(d['state_dict'])
        return cls(m,LearnedPositionValue(d['baseline_weights'],d['baseline_summary']),d['config'],
                   d['summary'],d['residual_scale'])
