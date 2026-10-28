import builtins
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from pai.thinker.predictive import *
from pai.thinker.hier_data import load_stream


def inputs():
    return torch.randn(2,8,19),torch.randn(2,8,4),torch.zeros(2,8,10)


def test_matched_budgets_and_refinement_no_parameter_gradients():
    h=HierarchicalPredictor();f=FlatPredictor(width=75)
    assert abs(parameters(h)/parameters(f)-1)<.1
    s,a,m=inputs()
    with torch.no_grad():pred,tr=h(s,a,m)
    assert pred.shape==s.shape and torch.isfinite(pred).all()
    assert all(p.grad is None for p in h.parameters())
    assert tr['iterations'].max()<=h.cfg.max_iterations


@pytest.mark.parametrize('kind',['hierarchical','flat','amortized'])
def test_self_supervised_gradient_updates(kind):
    torch.manual_seed(2)
    model=FlatPredictor(width=75) if kind=='flat' else HierarchicalPredictor(iterative=kind=='hierarchical')
    s,a,m=inputs();prediction,_=model(s,a,m)
    loss=predictive_loss(prediction,s.roll(-1,1),model);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_future_changes_cannot_change_past_predictions():
    torch.manual_seed(4);model=HierarchicalPredictor();s,a,m=inputs()
    with torch.no_grad():first,trace=model(s,a,m)
    changed=s.clone();changed[:,5:]+=100
    with torch.no_grad():second,_=model(changed,a,m)
    assert torch.equal(first[:,:5],second[:,:5])


def test_single_pass_ablation_and_three_distinct_timescales():
    model=HierarchicalPredictor(iterative=False);s,a,m=inputs()
    with torch.no_grad():_,trace=model(s,a,m)
    assert trace['iterations'].max()==1
    assert torch.equal(trace['level3'][:,0],trace['level3'][:,-1])
    assert not torch.equal(trace['level1'][:,0],trace['level1'][:,-1])


def test_stream_loader_rejects_truth_fields(tmp_path):
    p=tmp_path/'bad.npz'
    np.savez(p,state=np.zeros((3,19)),action=np.zeros((2,4)),context=np.zeros((2,10)),tokens=0,q=0,dq=0,label=1)
    with pytest.raises(ValueError):load_stream(p)


def test_rgb_perception_cannot_read_simulator_state():
    from pai.thinker.hier_sensors import OwnPixelPerception
    class Observation(dict):
        def __getitem__(self,key):
            assert key in ('ee_pos','gripper','wrist_force')
            return super().__getitem__(key)
    class Geometry:
        rest_z=np.array([.02,.01])
        def label(self,image):pass
        def masks(self,i):return np.ones((4,4),bool),np.zeros((4,4),bool)
        def on_plane(self,u,v,z):return np.array([.5,0,z])
    p=OwnPixelPerception.__new__(OwnPixelPerception)
    p.env=SimpleNamespace(render=lambda c:np.zeros((4,4,3),np.uint8),control_dt=.02)
    p.geom=Geometry();p.camera='given';p.specs=[SimpleNamespace(kind='block'),SimpleNamespace(kind='plate')]
    p.colours=[[1,0,0],[1,1,1]];p.reset()
    out=p(Observation(ee_pos=np.array([.5,0,.4]),gripper=.08,wrist_force=np.ones(3)))
    assert out.shape==(3,19) and np.isfinite(out).all()

def test_sensory_recording_preserves_default_agent_behavior():
    from pai.agents.slice_agent import run_episode
    from pai.config import load_config
    from pai.envs import TabletopEnv
    from pai.world.model import load_world_model
    cfg=load_config('configs/tabletop.yaml',['env.render_images=false','slice.samples=4',
        'slice.iterations=1','slice.horizon=2','slice.max_plan_steps=3','slice.settle_steps=0'])
    wm=load_world_model(cfg.slice.world_model,'cpu');env=TabletopEnv(cfg.env,disturbances=[])
    a=run_episode(env,wm,cfg,0,89700,device='cpu')
    senses=[];b=run_episode(env,wm,cfg,0,89700,device='cpu',sensory=senses)
    assert np.array_equal(a.tokens,b.tokens) and np.array_equal(a.actions,b.actions)
    assert len(senses)==len(b.actions)
    assert set(senses[0])=={'before','after','action','holding','phase','t','q','dq'}

def test_result_retry_is_scoped_and_preserves_content(tmp_path,monkeypatch):
    from pai.thinker.hier_io import install_replace_retry
    monkeypatch.chdir(tmp_path);root=tmp_path/'results';root.mkdir()
    src=root/'thinker_hier_test.json.tmp';dst=root/'thinker_hier_test.json';src.write_text('unchanged')
    real=Path.replace;calls=[]
    def transient(self,target):
        calls.append(1)
        if len(calls)==1:raise PermissionError('transient sharing violation')
        return real(self,target)
    monkeypatch.setattr(Path,'replace',transient)
    install_replace_retry()
    try:src.replace(dst)
    finally:Path.replace=real
    assert len(calls)==2 and dst.read_text()=='unchanged'


def test_scoring_censors_missing_adaptation_and_handles_constant_states():
    from pai.thinker.hier_analysis import episode_metrics,tau
    traces={k:np.zeros((30,3)) for k in ('level1','level2','level3')}
    traces.update(iterations=np.ones(30,int),surprise=np.arange(30),refinement_before=np.zeros(30),refinement_after=np.zeros(30))
    result=dict(error=np.full(30,10.),traces=traces,code_probabilities=None)
    truth=dict(disturbances=[dict(t=11)])
    metrics=episode_metrics(result,truth,clean_level=1.)
    assert metrics['onset']==2 and metrics['post20_mse']==10
    assert metrics['censored'] and metrics['adaptation_steps'] is None
    assert metrics['restricted_adaptation']==20
    assert tau(np.zeros((30,2)))['steps'] is None

def test_partial_post_onset_windows_are_not_scored_as_full20():
    from pai.thinker.hier_analysis import episode_metrics
    traces={k:np.arange(30)[:,None].astype(float) for k in ('level1','level2','level3')}
    traces.update(iterations=np.ones(30,int),surprise=np.arange(30),refinement_before=np.zeros(30),refinement_after=np.zeros(30))
    r=episode_metrics(dict(error=np.ones(30),traces=traces,code_probabilities=None),dict(disturbances=[dict(t=126)]),2.)
    assert r['post_available_steps']==5 and not r['complete_post20']
    assert r['post20_mse'] is None and r['restricted_adaptation'] is None
    assert r['post_available_mse']==1. and r['adaptation_steps']==2
