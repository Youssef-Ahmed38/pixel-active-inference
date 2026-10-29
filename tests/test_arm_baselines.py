import numpy as np
import torch

from pai.baselines.arm import observation,potential
from pai.baselines.ppo import ActorCritic,gae,loss


def test_sensor_features_and_reward_ignore_truth():
    tokens=np.zeros((7,19),np.float32);tokens[:,0]=.5;tokens[1,2]=.02;tokens[5,2]=.006;tokens[0,8]=.08
    x=observation(tokens,np.zeros(7),np.zeros(7),np.zeros(4),0)
    assert x.shape==(158,) and np.isfinite(x).all() and np.isfinite(potential(tokens))
    # Interface accepts only own tokens/proprioception, not a full truth-bearing observation.
    import inspect
    assert set(inspect.signature(observation).parameters)=={'tokens','q','dq','previous_action','elapsed'}


def test_timeout_bootstraps_but_episode_recursion_stops():
    reward=torch.tensor([[1.],[2.]])
    values=torch.zeros_like(reward);next_value=torch.tensor([[3.],[0.]])
    adv,_=gae(reward,values,next_value,torch.ones_like(reward))
    assert torch.allclose(adv,torch.tensor([[3.97],[2.]]))


def test_recurrent_reset_erases_previous_episode_memory():
    model=ActorCritic(recurrent=True);obs=torch.randn(3,158)
    a,v,h=model(obs,torch.randn(3,128),torch.ones(3))
    b,w,j=model(obs,torch.zeros(3,128),torch.zeros(3))
    assert torch.equal(a,b) and torch.equal(v,w) and torch.equal(h,j)


def test_squashed_logprob_and_clipped_losses_are_finite_with_gradients():
    for recurrent in (False,True):
        policy=ActorCritic(recurrent=recurrent);obs=torch.randn(8,158)
        a,lp,v,h,raw=policy.act(obs)
        assert (a.abs()<=1).all() and torch.isfinite(lp).all()
        if recurrent:
            objective,_=loss(policy,obs.reshape(4,2,158),raw.detach().reshape(4,2,4),lp.detach().reshape(4,2),torch.ones(4,2),torch.ones(4,2),torch.zeros(2,128),torch.zeros(4,2))
        else:objective,_=loss(policy,obs,raw.detach(),lp.detach(),torch.ones(8),torch.ones(8))
        objective.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in policy.parameters())


def test_paired_tests_use_identical_episodes_and_discordant_counts():
    from pai.baselines.statistics import paired
    import pytest
    full=[dict(index=i,seed=i,disturbance_draw_sha256='same',success=i<5) for i in range(10)]
    candidate=[dict(**{k:v for k,v in row.items() if k!='success'},success=False) for row in full]
    result=paired(candidate,full)
    assert result['full_only']==5 and result['candidate_only']==0 and result['two_sided_sign_p']==.0625
    candidate[0]['seed']=100
    with pytest.raises(ValueError):paired(candidate,full)


def test_worker_randomization_exactly_matches_slice_distribution():
    from pai.baselines.disturbances import disturbance_for
    from scripts.slice_eval import disturbance_for as original
    for condition in ('none','push','heavier_object','slippery_object','camera_shift'):
        for seed in (1,3,7):assert disturbance_for(condition,np.random.default_rng(seed))==original(condition,np.random.default_rng(seed))


def test_checkpoint_restores_optimizer_and_random_streams(tmp_path):
    import random
    from pai.baselines.checkpoint import atomic_save,random_state,restore_random
    torch.manual_seed(143);model=torch.nn.Linear(3,1);optimizer=torch.optim.Adam(model.parameters())
    def update():
        x=torch.randn(4,3);optimizer.zero_grad();model(x).square().sum().backward();optimizer.step()
    update();path=tmp_path/'resume.pt'
    atomic_save(path,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),random=random_state()))
    expected=(random.random(),np.random.rand());update();weights=[p.detach().clone() for p in model.parameters()]
    state=torch.load(path,weights_only=False);model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
    restore_random(state['random']);assert (random.random(),np.random.rand())==expected
    update();assert all(torch.equal(a,b) for a,b in zip(weights,model.parameters()))


def test_worker_checkpoint_replays_exactly_across_process_and_episode_reset():
    from pai.baselines.vector import Vector
    rng=np.random.default_rng(12);worker=Vector(1,98,'checkpoint_test')
    try:
        worker.reset()
        for _ in range(140):worker.step(rng.uniform(-1,1,(1,4)))
        state=worker.snapshot();actions=[rng.uniform(-1,1,(1,4)) for _ in range(15)]
        expected=[worker.step(a) for a in actions]
    finally:worker.close()
    worker=Vector(1,98,'checkpoint_test_restore')
    try:
        worker.restore(state)
        for action,(arrays,summaries) in zip(actions,expected):
            actual,score=worker.step(action)
            assert all(np.array_equal(a,b) for a,b in zip(arrays,actual))
            assert score==summaries
    finally:worker.close()


