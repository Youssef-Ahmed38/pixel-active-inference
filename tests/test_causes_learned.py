"""Fast checks of learned arm densities, evidence boundary and monitor adapter."""
from types import SimpleNamespace

import numpy as np
import pytest

from pai.causes.inference import CAUSES, Calibration, CauseReport, StepEvidence
from pai.causes.learned import (KNOWN, LearnedArmLikelihood, LearnedEpisodeMonitor,
                               extract_features, learned_episode_monitor)


def synthetic(cause='none', seed=0, n=20):
    rng = np.random.default_rng(seed)
    sig = np.r_[np.full(3,.001),np.full(3,.3),np.full(6,.001)]
    out=[]
    for t in range(n):
        r = rng.normal(size=12)*sig*.3
        if cause=='push' and 6<=t<12:
            r[[0,6]]+=.008;r[3]+=15.
        if cause=='heavier_object':
            r[[2,8]]-=.004;r[5]+=4.
        if cause=='slippery_object' and t>=8:
            r[8]-=.02;r[5]-=.5
        if cause=='camera_shift' and t==8:
            r[6:9]+=[.04,-.02,.005];r[9:12]+=[.04,-.02,.005]
        out.append(StepEvidence(t,r,sig.copy(),np.array([0.,0.,.1,-1.]),True,'lifted'))
    return out


def training():
    rows=[];labels=[]
    for c in KNOWN:
        for seed in range(4):
            rows.append(extract_features(synthetic(c,seed),triggered=c!='none')[0]);labels.append(c)
    return rows,labels


def test_fitted_densities_identify_physical_examples_and_keep_estimates():
    rows,labels=training()
    m=LearnedArmLikelihood.fit(rows,labels,shrinkage=1.)
    for c in KNOWN:
        report=m.infer(synthetic(c,6),triggered=c!='none')
        assert report.best==c,(c,report.posterior)
        assert sum(report.posterior.values())==pytest.approx(1)
    cam=m.infer(synthetic('camera_shift',6),triggered=True)
    assert np.allclose(cam.params['camera_shift']['camera_offset_mm'],[-40,20,-5],atol=2)
    mass=m.infer(synthetic('heavier_object',6),triggered=True)
    assert mass.params['heavier_object']['extra_mass_kg']==pytest.approx(4/9.81,abs=.03)


def test_teacher_label_permutation_changes_learned_association():
    rows,labels=training()
    mapping=dict(zip(KNOWN,KNOWN[1:]+KNOWN[:1]))
    m=LearnedArmLikelihood.fit(rows,[mapping[c] for c in labels],shrinkage=1.)
    for c in KNOWN:
        f,_=extract_features(synthetic(c,7),triggered=c!='none')
        p=m.posterior(f)
        assert max(p,key=p.get)==mapping[c]


def test_checkpoint_roundtrip_and_broad_unknown():
    rows,labels=training()
    m=LearnedArmLikelihood.fit(rows,labels,feature_set='raw',shrinkage=1.)
    restored=LearnedArmLikelihood.from_dict(m.to_dict())
    assert restored.posterior(rows[0])==pytest.approx(m.posterior(rows[0]))
    extreme={k:1e50 for k in m.names}
    p=restored.posterior(extreme)
    assert max(p,key=p.get)=='unknown'
    assert '_precisions' not in m.to_dict()


def test_features_ignore_added_truth_metadata_and_use_sensor_values():
    hist=synthetic('push',2)
    before,_=extract_features(hist,triggered=True)
    for e in hist:
        e.true_cause='camera_shift';e.event_log=[{'force':[0,999,0]}]
    after,_=extract_features(hist,triggered=True)
    assert before==after
    hist[8].residual=hist[8].residual.copy();hist[8].residual[3]+=10
    changed,_=extract_features(hist,triggered=True)
    assert changed!=before


def test_bad_evidence_and_training_labels_are_rejected():
    with pytest.raises(ValueError):extract_features([])
    hist=synthetic();hist[0].sigma[0]=0
    with pytest.raises(ValueError):extract_features(hist)
    rows,labels=training()
    with pytest.raises(ValueError):LearnedArmLikelihood.fit(rows,['none']*len(rows))
    with pytest.raises(ValueError):LearnedArmLikelihood.fit(rows,labels,weights=[0]*len(rows))


def test_missing_readout_withholds_control_without_changing_diagnosis():
    class NoMassEstimate:
        def infer(self,hist,dt,triggered):
            return CauseReport(0,hist[-1].t,{c:float(c=='heavier_object') for c in CAUSES},{'heavier_object':{}})
    m=LearnedEpisodeMonitor(.1,likelihood=NoMassEstimate(),online_likelihood=NoMassEstimate(),
                           calibration=Calibration(threshold=0.))
    assert m.add(synthetic('push')[0]) is None
    assert m.reports[-1].best=='heavier_object'
    assert m.reports[-1].posterior['heavier_object']==1.
    assert len(m.withheld_adaptations)==1


def test_scoped_adapter_restores_shared_agent_on_exception():
    from pai.agents import slice_agent
    original=slice_agent.SurpriseMonitor
    with pytest.raises(RuntimeError):
        with learned_episode_monitor(SimpleNamespace()):
            assert slice_agent.SurpriseMonitor is not original
            raise RuntimeError('test exit')
    assert slice_agent.SurpriseMonitor is original


def test_prefix_teaching_uses_only_events_that_have_already_started():
    from scripts.arm_causes import encode_history,prefix_rows
    hist=synthetic('push')
    r={'seed':1,'own':{'history':encode_history(hist),'trigger_steps':[7,18]},
       'scoring':{'disturbances':[{'t':60,'disturbance':'push','params':{}}]}}
    rows=prefix_rows([r],.1,5)
    assert [r['label'] for r in rows]==['none','none','push','push']
    assert sum(r['weight'] for r in rows)==pytest.approx(1.)
    assert not any('truth' in k or 'onset_time' in k for r in rows for k in r['features'])


def test_slip_onset_is_scored_against_matching_object_slip_not_friction_start():
    from scripts.arm_causes import parameter_errors
    row={'scoring':{'cause':'slippery_object','disturbances':[{'t':1,'disturbance':'friction','params':{}}],
                    'events':[{'t':2,'type':'slip','object':'green'},{'t':15,'type':'slip','object':'red'}]}}
    e=parameter_errors(row,'slippery_object',{'slippery_object':{'onset':3}},.1,.02)
    assert e['onset_error_s']==pytest.approx(0.)
    assert parameter_errors(row,'push',{'push':{'onset':0}},.1,.02)=={}


def test_gzip_payload_roundtrip_and_oversize_guard(tmp_path):
    from scripts.arm_causes import save,load
    p=tmp_path/'own.json.gz';save(p,{'history':[1,2,3]});assert load(p)=={'history':[1,2,3]}
    with pytest.raises(FileExistsError):save(p,{})
    with pytest.raises(ValueError):save(tmp_path/'oversize.json',{'x':'a'*5_000_001})


def test_trace_roundtrip_preserves_numeric_dtypes_for_analytic_readouts():
    from scripts.arm_causes import encode_history,history
    hist=synthetic('push')
    for e in hist:
        e.residual=e.residual.astype(np.float32);e.action=e.action.astype(np.float32)
    restored=history({'own':{'history':encode_history(hist)}})
    for a,b in zip(hist,restored):
        assert a.residual.dtype==b.residual.dtype and a.sigma.dtype==b.sigma.dtype
        assert np.array_equal(a.residual,b.residual) and np.array_equal(a.sigma,b.sigma)


@pytest.mark.parametrize('prior_kind',['uniform','legacy'])
def test_heavy_tailed_density_is_learned_and_checkpointed(prior_kind):
    rows,labels=training()
    model=LearnedArmLikelihood.fit(rows,labels,feature_set='raw',shrinkage=.9,
                                 likelihood_family='student_t4',prior_kind=prior_kind)
    restored=LearnedArmLikelihood.from_dict(model.to_dict())
    assert restored.likelihood_family=='student_t4' and restored.prior_kind==prior_kind
    for cause in KNOWN:
        f,_=extract_features(synthetic(cause,9),triggered=cause!='none')
        posterior=restored.posterior(f)
        assert max(posterior,key=posterior.get)==cause
        assert sum(posterior.values())==pytest.approx(1.)
    with pytest.raises(ValueError):
        LearnedArmLikelihood.from_dict({**model.to_dict(),'likelihood_family':'bad'})
