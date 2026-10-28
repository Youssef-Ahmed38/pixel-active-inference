from types import SimpleNamespace
import numpy as np
import pytest
import torch
from pai.locomotion.env import OBS_DIM,ACT_DIM,EnvConfig,LocomotionEnv
from pai.locomotion.body import load_body
from pai.locomotion.falls import SurpriseMonitor
from pai.locomotion.fall_causes import (CHANNELS,instrumented_body,load_cells,FallEvidenceMonitor,
    extract_features,pack_trace,unpack_trace,LearnedFallLikelihood)


class Still:
    out_norm=SimpleNamespace(std=torch.ones(33))
    def predict(self,s,a): return s
    def surprise_and_disagreement(self,s,a,s2):
        return ((s2-s)**2).mean(-1),torch.zeros(len(s))


def test_early_impulse_survives_old_ring_buffer_and_loaded_slip_uses_force():
    model=Still()
    mon=FallEvidenceMonitor(model,1)
    old=SurpriseMonitor(model,1,window=100)
    obs=np.zeros((1,OBS_DIM),np.float32)
    obs[:,5]=-1.
    h=np.zeros((1,2))
    v=np.zeros((1,2,3))
    v[:,:,0]=1.
    force=np.zeros((1,2,3))
    for t in range(180):
        nxt=obs.copy()
        if t==30: nxt[0,6]+=1.
        a=np.zeros((1,ACT_DIM),np.float32)
        mon.update(obs,a,nxt,h,v,force,np.zeros_like(force))
        old.update(obs,a,nxt,h,v)
        obs=nxt
    np.testing.assert_allclose(mon.legacy.features([0]),old.features([0]))
    assert old.features([0])[0,0]==0.
    features,_=extract_features(mon.history[0])
    assert features['all_impulse_10']==pytest.approx(1.)
    assert features['all_loaded_sliding_p90']==0.
    assert old.features([0])[0,1]==pytest.approx(1.)
    mon.reset([0])
    assert not mon.history[0]
    np.testing.assert_array_equal(mon.legacy.features([0]),np.zeros((1,2)))


def test_sensor_trace_roundtrip_and_invalid_values():
    trace=np.random.default_rng(1).normal(size=(100,len(CHANNELS))).astype(np.float32)
    np.testing.assert_array_equal(unpack_trace(pack_trace(trace)),trace)
    trace[0,0]=np.nan
    with pytest.raises(ValueError): extract_features(trace)


def test_learned_distributions_change_with_labels_and_unknown_catchall():
    rng=np.random.default_rng(2)
    features=[dict(a=float(np.exp(rng.normal(mu,.05))),b=float(np.exp(rng.normal(-mu,.05))))
              for mu in (-2.,0.,2.) for _ in range(30)]
    labels=np.repeat(['gait','push','slip'],30)
    model=LearnedFallLikelihood.fit(features,labels)
    p=model.posterior([features[0],features[40],features[80]])
    assert p.argmax(1).tolist()==[0,1,2]
    restored=LearnedFallLikelihood(**model.to_dict())
    np.testing.assert_allclose(p,restored.posterior([features[0],features[40],features[80]]))
    changed=LearnedFallLikelihood.fit(features,np.repeat(['slip','push','gait'],30))
    assert changed.posterior([features[0]]).argmax()==2
    assert model.posterior([dict(a=1e30,b=1e30)]).argmax()==3
    np.testing.assert_allclose(p.sum(1),1.)


def test_read_only_sensor_instrumentation_preserves_physics():
    ordinary=load_body()
    sensed,addresses=instrumented_body()
    for field in ('body_mass','body_inertia','pair_friction','actuator_gainprm'):
        np.testing.assert_array_equal(getattr(ordinary.model,field),getattr(sensed.model,field))
    assert (ordinary.model.nq,ordinary.model.nv,ordinary.model.nu)==(sensed.model.nq,sensed.model.nv,sensed.model.nu)
    cfg=EnvConfig(randomize=False,nthread=1,obs_noise=0.)
    a=LocomotionEnv(1,cfg,seed=9,body=ordinary)
    b=LocomotionEnv(1,cfg,seed=9,body=sensed)
    for _ in range(5):
        action=np.zeros((1,ACT_DIM))
        a.step(action)
        b.step(action)
        np.testing.assert_allclose(a.qpos(),b.qpos(),atol=1e-12)
    f,t=load_cells(b.sensors,addresses)
    assert f.shape==t.shape==(1,2,3)
    assert np.isfinite(f).all() and np.linalg.norm(f)>0.


def test_learned_likelihood_learns_the_causes_present_in_its_falls():
    """A model fitted on gait/push/slip/wall falls has four causes plus the catch-all; old
    three-cause checkpoints (no 'causes' key) still load as gait/push/slip."""
    import numpy as np
    from pai.locomotion.fall_causes import LearnedFallLikelihood

    rng = np.random.default_rng(0)
    centres = {"gait": 0.0, "push": 3.0, "slip": -3.0, "wall": 6.0}
    feats, labels = [], []
    for cause, c in centres.items():
        for _ in range(20):
            feats.append({"a": float(np.exp(c / 4 + rng.normal(0, .1))), "b": float(rng.random())})
            labels.append(cause)
    model = LearnedFallLikelihood.fit(feats, labels)
    assert model.labels == ("gait", "push", "slip", "wall", "unknown")
    pred = [model.labels[j] for j in model.posterior(feats).argmax(1)]
    assert np.mean([p == t for p, t in zip(pred, labels)]) > 0.9
    old = model.to_dict()
    old.pop("causes")
    old["means"], old["covariances"] = old["means"][:3], old["covariances"][:3]
    assert LearnedFallLikelihood(**old).labels == ("gait", "push", "slip", "unknown")
