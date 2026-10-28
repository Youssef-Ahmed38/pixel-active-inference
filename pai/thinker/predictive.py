"""Three-timescale sensory predictor and a size-matched flat recurrent control.

No cause labels, event onsets, teacher posteriors, or simulator imports.
Inner inference explains a completed past transition, never the future target.
"""
from dataclasses import dataclass
import torch
from torch import nn

@dataclass
class PredictiveConfig:
    senses:int=19
    action:int=4
    motor:int=10
    fast:int=48
    latent:int=16
    codes:int=8
    slow:int=16
    context_period:int=4
    episode_period:int=16
    max_iterations:int=5
    inference_lr:float=.15
    difficulty_threshold:float=1.

class HierarchicalPredictor(nn.Module):
    def __init__(self,cfg=None,iterative=True):
        super().__init__();self.cfg=cfg or PredictiveConfig();c=self.cfg;self.iterative=iterative
        self.fast=nn.GRUCell(c.senses+c.action+c.motor+c.latent,c.fast)
        self.prior=nn.Sequential(nn.Linear(c.fast+c.latent+c.slow+c.senses,64),nn.Tanh(),nn.Linear(64,c.latent))
        self.slow=nn.GRUCell(c.fast+c.motor,c.slow)
        self.head=nn.Sequential(nn.Linear(c.fast+c.latent+c.senses,64),nn.Tanh(),nn.Linear(64,c.senses))
        self.log_variance=nn.Parameter(torch.zeros(c.senses))
    def latent(self,z):
        k=self.cfg.codes
        return torch.cat((z[:,:k].softmax(-1),torch.tanh(z[:,k:])),dim=-1)
    def initial(self,batch,device):
        c=self.cfg
        return dict(fast=torch.zeros(batch,c.fast,device=device),z=torch.zeros(batch,c.latent,device=device),
                    slow=torch.zeros(batch,c.slow,device=device),previous=None,t=0)
    def predict(self,s,a,m,h,z):
        context=self.latent(z)
        h=self.fast(torch.cat((s,a,m,context),-1),h)
        return s+self.head(torch.cat((h,context,s),-1)),h
    def step(self,s,a,m,state,refine=None):
        c=self.cfg;t=state['t'];h=state['fast'];slow=state['slow'];z=state['z'];prev=state['previous']
        error=torch.zeros_like(s) if prev is None else s-prev['prediction'].detach()
        surprise=(error.square()*self.log_variance.detach().clamp(-6,4).neg().exp()).mean(-1)
        if t%c.episode_period==0:
            slow=self.slow(torch.cat((h,m),-1),slow)
        iterations=torch.zeros(len(s),device=s.device,dtype=torch.long)
        before=torch.zeros(len(s),device=s.device);after=before.clone()
        update_mask=torch.ones(len(s),device=s.device,dtype=torch.bool) if t%c.context_period==0 else surprise>2*c.difficulty_threshold
        update=bool(update_mask.any())
        if update:
            prior=self.prior(torch.cat((h,z,slow,error),-1))
            chosen=prior
            use=self.iterative if refine is None else refine
            if use and prev is not None:
                # Synchronous batch updates, with genuinely inactive examples
                # dropped from the inner objective once their budget is spent.
                budgets=(1+(surprise/c.difficulty_threshold).sqrt().long()).clamp(1,c.max_iterations)*update_mask
                with torch.enable_grad():
                    chosen=prior.detach().requires_grad_(True)
                    for j in range(c.max_iterations):
                        active=budgets>j
                        if not active.any():break
                        ids=active.nonzero().flatten()
                        p,_=self.predict(prev['s'][ids].detach(),prev['a'][ids].detach(),prev['m'][ids].detach(),prev['h'][ids].detach(),chosen[ids])
                        losses=((p-s[ids].detach()).square()*self.log_variance.detach().clamp(-6,4).neg().exp()).mean(-1)
                        objective=losses+.01*(chosen[ids]-prior[ids].detach()).square().mean(-1)
                        grad=torch.autograd.grad(objective.sum(),chosen,create_graph=False)[0]
                        if j==0:before[ids]=losses.detach()
                        chosen=(chosen-c.inference_lr*grad).detach().requires_grad_(True)
                        iterations[ids]+=1
                    p,_=self.predict(prev['s'].detach(),prev['a'].detach(),prev['m'].detach(),prev['h'].detach(),chosen)
                    after=((p-s.detach()).square()*self.log_variance.detach().clamp(-6,4).neg().exp()).mean(-1).detach()
                # Straight-through first-order inference: no second-order inner
                # gradients. Both branches get the same prediction loss budget.
                chosen=prior+(chosen.detach()-prior.detach())
            else:iterations[update_mask]=1
            z=torch.where(update_mask[:,None],chosen,z)
        pred,new_h=self.predict(s,a,m,h,z)
        next_state=dict(fast=new_h,z=z,slow=slow,t=t+1,
                        previous=dict(s=s,a=a,m=m,h=h,prediction=pred))
        return pred,next_state,dict(level1=new_h,level2=self.latent(z),level3=slow,
              surprise=surprise,iterations=iterations,refinement_before=before,refinement_after=after)
    def forward(self,s,a,m,refine=None):
        state=self.initial(len(s),s.device);predictions=[];traces=[]
        for t in range(s.shape[1]):
            p,state,tr=self.step(s[:,t],a[:,t],m[:,t],state,refine)
            predictions.append(p);traces.append(tr)
        return torch.stack(predictions,1),{k:torch.stack([v[k] for v in traces],1) for k in traces[0]}

class FlatPredictor(nn.Module):
    def __init__(self,cfg=None,width=80):
        super().__init__();self.cfg=cfg or PredictiveConfig();c=self.cfg;self.width=width
        self.cell=nn.GRUCell(c.senses+c.action+c.motor,width)
        self.head=nn.Sequential(nn.Linear(width+c.senses,64),nn.Tanh(),nn.Linear(64,c.senses))
        self.log_variance=nn.Parameter(torch.zeros(c.senses))
    def forward(self,s,a,m,refine=None):
        h=torch.zeros(len(s),self.width,device=s.device);outputs=[];states=[]
        for t in range(s.shape[1]):
            h=self.cell(torch.cat((s[:,t],a[:,t],m[:,t]),-1),h)
            outputs.append(s[:,t]+self.head(torch.cat((h,s[:,t]),-1)));states.append(h)
        states=torch.stack(states,1);prediction=torch.stack(outputs,1)
        previous=torch.cat((s[:,:1],prediction[:,:-1].detach()),1)
        surprise=((s-previous).square()*self.log_variance.detach().clamp(-6,4).neg().exp()).mean(-1)
        return prediction,dict(level1=states,level2=states,level3=states,
            surprise=surprise,iterations=torch.ones(s.shape[:2],device=s.device,dtype=torch.long),
            refinement_before=torch.zeros_like(surprise),refinement_after=torch.zeros_like(surprise))

def parameters(model):return sum(p.numel() for p in model.parameters())
def matched_flat(cfg=None):
    cfg=cfg or PredictiveConfig();target=parameters(HierarchicalPredictor(cfg))
    width=min(range(16,129),key=lambda w:abs(parameters(FlatPredictor(cfg,w))-target))
    return FlatPredictor(cfg,width)

def predictive_loss(prediction,target,model):
    logvar=model.log_variance.clamp(-6,4)
    return .5*((prediction-target).square()*(-logvar).exp()+logvar).mean()
