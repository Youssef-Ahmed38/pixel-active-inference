import copy
import numpy as np
import pytest
from pai.language.association import EvidenceLexicon,QuestionIntent,BeliefCoordinates
from scripts.language_causes_v2_eval import instruction_model,GOALS,score_v2
from pai.language.causal_teacher import PHRASES
from pai.language.teacher import utterance
from pai.language.association_teacher import question_rows,FINAL_QUESTIONS
from language_causes_v2_summary import rate


def pairs():
    return [dict(phrase=phrase,posterior=list(p),features=[[v],[100*v]],
                 numbers=[dict(template='amount {value}',value=[v])])
            for phrase,p in [('opaque word A',[1.,0.]),('opaque word B',[0.,1.])]
            for v in (1.,2.,3.,4.)]


def test_observations_determine_mapping_and_parameter_binding():
    rows=pairs()
    model=EvidenceLexicon.fit(rows)
    assert model.phrase_by_coordinate=={'0':'opaque word A','1':'opaque word B'}
    assert all(m['feature']==0 for m in model.numeric)
    swapped=copy.deepcopy(rows)
    for r in swapped:
        r['posterior'].reverse()
        r['features'].reverse()
    other=EvidenceLexicon.fit(swapped)
    assert other.phrase_by_coordinate=={'1':'opaque word A','0':'opaque word B'}
    assert all(m['feature']==1 for m in other.numeric)


def test_uninformative_or_conflicting_beliefs_abstain():
    rows=pairs()
    for r in rows:
        r['posterior']=[.5,.5]
    assert not EvidenceLexicon.fit(rows).phrase_by_coordinate
    rows=pairs()
    for r in rows:
        r['posterior']=[0.,0.]
    assert not EvidenceLexicon.fit(rows).phrase_by_coordinate


def test_question_generalization_and_unclear_intent():
    q=QuestionIntent.fit([('cause why happened','why'),('why explain cause','why'),
                         ('open fingers early','open'),('early open hand','open')])
    assert q.predict('explain happened cause')=='why'
    assert q.predict('fingers hand early')=='open'
    assert q.predict('cause early') is None
    assert q.predict('unrelated words') is None


@pytest.mark.parametrize('language',['english','arabic'])
def test_all_varied_goals_are_compositional(language):
    model,n=instruction_model(language)
    assert n>0
    for goal in GOALS:
        assert model.parse(utterance(goal,language))==goal
    taught={text.casefold() for text,_ in question_rows(language)[0]}
    assert not taught & {text.casefold() for text,_ in FINAL_QUESTIONS[language]}


def test_onset_matches_inferred_event_not_last_other_event():
    out=dict(agent_cause='push',phrase=PHRASES['english']['cause:push'],goal_observed=False,
        numeric_claims=[dict(template=PHRASES['english']['estimate:onset'],value=[20.])])
    truth=dict(success=False,disturbances=[dict(disturbance='push',t=100,params={}),
                                         dict(disturbance='friction',t=1,params={})])
    score=score_v2(out,truth,'english')
    assert score['parameters'][0]['error']==0.
    assert score['parameters'][0]['passed']
    truth['disturbances']=truth['disturbances'][1:]
    score=score_v2(out,truth,'english')
    assert not score['parameters'][0]['event_matched']
    assert score['parameters'][0]['error'] is None


def test_wilson_intervals_preserve_extreme_uncertainty():
    lower=rate(0,10)
    upper=rate(10,10)
    assert lower['wilson95'][0]==0.
    assert lower['wilson95'][1]==pytest.approx(.2775328,abs=1e-6)
    assert upper['wilson95'][0]==pytest.approx(.7224672,abs=1e-6)
    assert upper['wilson95'][1]==pytest.approx(1.)
    assert rate(0,0)['wilson95'] is None


def test_grading_uses_spoken_meaning_not_latent_variable_name():
    out=dict(agent_cause='heavier_object',phrase=PHRASES['english']['cause:push'],
             goal_observed=True,numeric_claims=[])
    truth=dict(success=True,disturbances=[dict(disturbance='payload',t=1,params={'mass':.4})])
    score=score_v2(out,truth,'english')
    assert score['reported_cause']=='push'
    assert not score['cause_correct']
