"""Clipped PPO and recurrent clipped PPO; tanh-Gaussian Cartesian actions."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, obs_dim=158, recurrent=False):
        super().__init__(); self.recurrent = recurrent
        self.encoder = nn.Sequential(nn.Linear(obs_dim,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh())
        self.gru = nn.GRUCell(128,128) if recurrent else None
        self.actor = nn.Linear(128,4); self.critic = nn.Linear(128,1)
        self.log_std = nn.Parameter(torch.full((4,),-.5))
        for layer in self.modules():
            if isinstance(layer,nn.Linear):
                nn.init.orthogonal_(layer.weight,np.sqrt(2));nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.weight,.01);nn.init.orthogonal_(self.critic.weight,1.)

    def forward(self, obs, hidden=None, reset=None):
        x = self.encoder(obs)
        if self.recurrent:
            if hidden is None:hidden = torch.zeros_like(x)
            if reset is not None:hidden = hidden*(1-reset[:,None])
            hidden = self.gru(x,hidden); x = hidden
        return self.actor(x), self.critic(x).squeeze(-1), hidden

    def act(self, obs, hidden=None, reset=None, deterministic=False):
        mean,value,hidden = self(obs,hidden,reset)
        dist = Normal(mean,self.log_std.clamp(-5,1).exp())
        raw = mean if deterministic else dist.sample()
        action = raw.tanh()
        logprob = self.logprob(dist,raw)
        return action,logprob,value,hidden,raw

    @staticmethod
    def logprob(dist, raw):
        # Stable log(1-tanh(x)^2), including saturated actions.
        jac = 2*(np.log(2)-raw-torch.nn.functional.softplus(-2*raw))
        return (dist.log_prob(raw)-jac).sum(-1)


def gae(rewards, values, next_values, done, gamma=.99, lam=.95):
    advantage = torch.zeros_like(rewards); running = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t]+gamma*next_values[t]-values[t]
        running = delta+gamma*lam*(1-done[t])*running
        advantage[t] = running
    return advantage, advantage+values


def loss(policy, obs, raw, old_logprob, advantages, returns, hidden=None, reset=None):
    if obs.ndim==3:
        means=[];values=[]
        for t in range(len(obs)):
            m,v,hidden=policy(obs[t],hidden,reset[t]);means.append(m);values.append(v)
        mean,value=torch.stack(means),torch.stack(values)
    else:mean,value,_=policy(obs)
    dist=Normal(mean,policy.log_std.clamp(-5,1).exp())
    logprob=policy.logprob(dist,raw)
    ratio=(logprob-old_logprob).exp()
    actor=-torch.minimum(ratio*advantages,ratio.clamp(.8,1.2)*advantages).mean()
    critic=.5*(value-returns).square().mean()
    entropy=dist.entropy().sum(-1).mean()
    return actor+.5*critic-.01*entropy, dict(approx_kl=float(((ratio-1)-(logprob-old_logprob)).mean().detach()))
