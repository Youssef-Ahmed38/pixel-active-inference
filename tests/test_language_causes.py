import pytest
from pai.language.causal import ClauseLanguage, CausalExplainer, EpisodeEvidence
from pai.language.causal_teacher import teaching_rows, episode_teaching_rows


def evidence(**kw):
    data = dict(episode=1, posterior={'heavier_object': .99, 'none': .01},
                parameters={'heavier_object': {'extra_mass_kg': .431}},
                adaptations=[], fallbacks=0, goal_observed=True, credits=[])
    return EpisodeEvidence(**(data | kw))


@pytest.mark.parametrize('language',['english','arabic'])
def test_numeric_generalization_and_independent_teaching(language):
    model = ClauseLanguage.fit(teaching_rows(language))
    out = CausalExplainer(model).explain(evidence())
    assert '0.431' in out['text'] and out['cause'] == 'heavier_object'
    assert out['goal_observed'] is True
    # Changing a teacher phrase changes inference text, not semantics.
    rows = teaching_rows(language)
    for row in rows:
        if row['meaning'] == 'cause:heavier_object':
            row['text'] = 'NEWCAUSE'
    assert 'NEWCAUSE' in CausalExplainer(ClauseLanguage.fit(rows)).explain(evidence())['text']


@pytest.mark.parametrize('posterior',[{}, {'push':float('nan')}, {'push':2},
                                     {'push':.5,'none':.5}, {'unknown':1}, {'push':.8,'none':.2}])
def test_weak_or_invalid_evidence_abstains(posterior):
    out = CausalExplainer(ClauseLanguage.fit(teaching_rows('english'))).explain(evidence(posterior=posterior))
    assert out['abstained'] and not out['estimates']


def test_truth_fields_are_never_read():
    class Record:
        episode=1
        cause_posterior={'push':1.}
        cause_params={}
        adaptations=[]
        fallbacks=0
        def __getattr__(self,name):
            raise AssertionError('Forbidden read: '+name)
    ev = EpisodeEvidence.from_record(Record())
    assert ev.goal_observed is None
    assert CausalExplainer(ClauseLanguage.fit(teaching_rows('english'))).explain(ev)['cause']=='push'


@pytest.mark.parametrize('language',['english','arabic'])
def test_counterfactual_validity_and_false_drop_presupposition(language):
    model = ClauseLanguage.fit(teaching_rows(language))
    ex = CausalExplainer(model)
    ev = evidence(credits=[dict(name='open_early',valid=True,necessary=True,goal_cost=1.,baseline_cost=0.,phase='lifted')])
    assert not ex.answer(model.say('q:cf:open_early'),ev)['abstained']
    assert ex.answer(model.say('q:cf:remove_push'),ev)['abstained']
    assert ex.answer(model.say('q:drop'),ev)['abstained']
    assert ex.answer('unseen nonsense',ev)['abstained']
    ev.credits[0]['valid']=False
    assert ex.answer(model.say('q:cf:open_early'),ev)['abstained']


@pytest.mark.parametrize('language',['english','arabic'])
def test_episode_question_slot_and_missing_cause(language):
    model = ClauseLanguage.fit(episode_teaching_rows(language, [dict(mode='baseline',seed=1,
                    condition='none',truth={'disturbances':[]},belief={})]))
    ex = CausalExplainer(model)
    ev = evidence(episode=6000000)
    question = model.say('q:why:episode',episode=ev.episode)
    assert ex.answer(question,ev)['abstained']  # heavier word was never taught
    ev.posterior={'none':1.}
    assert not ex.answer(question,ev)['abstained']
    assert ex.answer(model.say('q:why:episode',episode=6000001),ev)['abstained']


def test_invalid_estimates_and_credit_do_not_become_claims():
    model = ClauseLanguage.fit(teaching_rows('english'))
    ex = CausalExplainer(model)
    ev = evidence(parameters={'heavier_object':{'extra_mass_kg':float('nan')}},
                  credits=[dict(name='open_early',valid=True,necessary=True,
                                goal_cost=float('inf'),baseline_cost=0.)])
    assert not ex.explain(ev)['estimates']
    assert ex.answer(model.say('q:cf:open_early'),ev)['abstained']
    ev.credits = [dict(name='open_early',valid=True,necessary=True,goal_cost=1.,baseline_cost=0.)]*2
    assert ex.answer(model.say('q:cf:open_early'),ev)['abstained']


def test_actual_adaptation_can_be_reported_despite_uncertain_final_cause():
    ex = CausalExplainer(ClauseLanguage.fit(teaching_rows('english')))
    out = ex.explain(evidence(posterior={},goal_observed=False,
                             adaptations=[{'action':'recalibrated the camera'}]))
    assert out['abstained']
    assert 'I recalibrated my camera.' in out['text']
    assert 'does not show the goal reached' in out['text']
