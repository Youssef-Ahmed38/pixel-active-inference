import copy
import json
from pathlib import Path
import pytest
from pai.language.causal import EpisodeEvidence
from pai.language.causal_v4 import CalibratedSilence,LearnedComposition,load_v4
from pai.language.causal_v4_teacher import teacher,probability_rows
from pai.language.causal import ClauseLanguage


def calibrated(none_silent=18,disturbed_silent=1):
    return CalibratedSilence({c:dict(n=20,silent=none_silent if c=='none' else disturbed_silent,
                                    explicit_none=2 if c=='none' else 0)
                             for c in ('none','push','heavier_object','slippery_object','camera_shift')})


def checkpoint():
    d=json.loads(Path('results/language_causes_v3_checkpoint.json').read_text(encoding='utf8'))
    languages={}
    for lang in d['models']:
        t=teacher(lang)
        languages[lang]=dict(composition=LearnedComposition.fit(t['pairs']).__dict__,phrases=t['phrases'],
                            hedge=t['hedge'],replacements=t['replacements'],teacher_pairs=t['pairs'],
                            probability_frame=ClauseLanguage.fit(probability_rows(lang)).templates['probability'])
    d['v4']=dict(silence=calibrated().__dict__,languages=languages)
    return d


def evidence(**kwargs):
    return EpisodeEvidence(**(dict(episode=1,posterior={},parameters={},adaptations=[],fallbacks=0,
                                   goal_observed=True,credits=[])|kwargs))


def test_frames_induced_without_clause_labels():
    pairs=[('opaqueA','PREFIX opaqueA END'),('opaqueB','OTHER opaqueB END')]
    m=LearnedComposition.fit(pairs)
    assert m.say('opaqueA')=='PREFIX opaqueA END'
    assert m.say('opaqueB')=='OTHER opaqueB END'
    assert m.say('unknown') is None
    assert LearnedComposition.fit([(a,b.replace('PREFIX','NEW')) for a,b in pairs]).say('opaqueA')=='NEW opaqueA END'
    with pytest.raises(ValueError):
        LearnedComposition.fit([('x','x x')])


def test_every_producible_arabic_none_hedge_uses_verb_complementizer():
    t=teacher('arabic');m=LearnedComposition.fit(t['pairs'])
    for phrase in (t['phrases'][next(k for k in t['phrases'] if k.startswith('لم'))],t['hedge']):
        sentence=m.say(phrase)
        assert 'أنني لم ' in sentence
        assert 'أن لم ' not in sentence
    assert 'سببًا' in m.say(t['pairs'][0][0])
    assert 'شيئًا' in m.say(t['hedge']) and 'اضطرابًا صغيرًا' in m.say(t['hedge'])


def test_all_teacher_pairs_reproduce_and_agreement_is_preserved():
    for lang in ('english','arabic'):
        t=teacher(lang);m=LearnedComposition.fit(t['pairs'])
        assert all(m.say(p)==s for p,s in t['pairs'])
    arabic=teacher('arabic')
    assert any('المهمة تحققت' in s for _,s in arabic['pairs'])
    assert any('الهدف تحقق' in s for _,s in arabic['pairs'])
    assert any('الأجسام الأثقل تحركت' in s for _,s in arabic['pairs'])


def test_per_type_prior_prevents_zero_risk_and_pooling_overconfidence():
    m=calibrated(18,0)
    assert all(0<v<1 for v in m.diagnostics()['likelihoods'].values())
    assert m.posterior()['none']==pytest.approx(18.5/(18.5+4*.5))
    changed=calibrated(18,0);changed.counts['slippery_object']['silent']=10
    assert changed.posterior()['none']<m.posterior()['none']
    assert sum(changed.posterior().values())==pytest.approx(1)


@pytest.mark.parametrize('language',['english','arabic'])
def test_strength_matches_probability_and_truth_cannot_enter(language):
    d=checkpoint();models=load_v4(d);out=models[language].explain(evidence())
    assert out['claim_strength']=='hedged'
    assert out['confidence']==pytest.approx(18.5/24.5)
    assert out['none_probability']==out['confidence']
    assert '75.5' in out['text']
    d['v4']['silence']['counts']={c:dict(n=200,silent=200 if c=='none' else 0,explicit_none=0)
                                for c in d['v4']['silence']['counts']}
    high=load_v4(d)[language].explain(evidence())
    assert high['claim_strength']=='confident' and not high['abstained']
    d['v4']['silence']['counts']={c:dict(n=20,silent=10,explicit_none=10) for c in d['v4']['silence']['counts']}
    low=load_v4(d)[language].explain(evidence())
    assert low['abstained'] and low['agent_cause'] is None


@pytest.mark.parametrize('language',['english','arabic'])
def test_disturbed_english_unchanged_and_arabic_grammatical(language):
    model=load_v4(checkpoint())[language]
    ev=evidence(posterior={'push':1.},adaptations=[{'action':'expect the extra weight'}],fallbacks=3)
    out=model.explain(ev)
    if language=='english':
        assert out['text']==model.base.explain(ev)['text']
    else:
        assert 'أنّ قوةً خارجيةً دفعت ذراعي' in out['text']
        assert 'حدّثت' in out['text']
        assert '3 مرة' not in out['text']
        assert 'عدد مرات إعادة محاولة الإمساك: 3.' in out['text']


def test_calibration_fit_reads_belief_only_after_teacher_provides_label():
    rows=[dict(belief=dict(posterior={}),label='none'),dict(belief=dict(posterior={'push':1}),label='push')]
    m=CalibratedSilence.fit(rows,lambda r:r['label'])
    assert m.counts['none']['silent']==1
    assert m.counts['push']['silent']==0

def test_every_generated_arabic_sentence_matches_reviewed_inventory():
    import re
    from scripts.language_causes_v4_report import arabic_inventory
    d=checkpoint();model=load_v4(d)['arabic']
    inventory=arabic_inventory(d)
    patterns=[]
    for row in inventory:
        for sentence in row['text'].split('. '):
            patterns.append(re.compile('^'+re.escape(sentence.rstrip('.')).replace(re.escape('{value}'),'.+?')+'$'))
    for cause in (*d['coordinates']['causes'],None):
        for goal in (True,False,None):
            for count in (0,1,2,3,11,100):
                ev=evidence(posterior={} if cause is None else {cause:1.},goal_observed=goal,fallbacks=count,
                            adaptations=[{'action':k[7:]} for k in d['models']['arabic']['templates'] if k.startswith('action:')],
                            parameters={c:dict(extra_mass_kg=.42,camera_offset_mm=[-40.,20.,1.],onset=8.)
                                        for c in d['coordinates']['causes']})
                text=model.explain(ev)['text']
                assert 'أن لم ' not in text and 'أنني لم ' in text if cause is None else 'أن لم ' not in text
                for sentence in text.split('. '):
                    assert any(p.fullmatch(sentence.rstrip('.')) for p in patterns),sentence

def test_reliability_brier_and_empty_bins():
    from scripts.language_causes_v4_report import reliability
    r=reliability([0.,.5,1.],[False,True,True])
    assert r['brier']==pytest.approx(.25/3)
    assert sum(b['n'] for b in r['bins'])==3
    assert r['bins'][-1]['observed_accuracy']['k']==1
    assert reliability([],[])['brier'] is None


def test_invalid_monitor_probabilities_cannot_trigger_calibrated_none():
    model=load_v4(checkpoint())['arabic']
    for p in ({'none':float('nan')},{'none':2.},{'none':.4}):
        out=model.explain(evidence(posterior=p))
        assert out['abstained']
        assert 'none_probability' not in out

def test_scoring_decodes_actual_surface_not_old_phrase_or_cause_key():
    from scripts.language_causes_v4_score import score_v4
    from pai.language.causal_teacher import PHRASES
    model=load_v4(checkpoint())['arabic']
    output=model.explain(evidence(posterior={'push':1.}))
    output['phrase']=PHRASES['arabic']['cause:heavier_object']
    output['agent_cause']='heavier_object'
    truth=dict(success=True,disturbances=[dict(disturbance='push',t=1,params={})])
    assert score_v4(output,truth,'arabic')['cause_correct']
    truth['disturbances']=[dict(disturbance='payload',t=1,params={'mass':.4})]
    assert not score_v4(output,truth,'arabic')['cause_correct']
    output['surface_phrase']=teacher('arabic')['phrases'][PHRASES['arabic']['cause:heavier_object']]
    assert not score_v4(output,truth,'arabic')['cause_correct']

def test_all_arabic_teacher_and_probability_strings_survive_utf8():
    import re
    t=teacher('arabic')
    strings=[*t['phrases'].values(),t['hedge'],*t['replacements'].values(),
             *(s for pair in t['pairs'] for s in pair),*(r['text'] for r in probability_rows('arabic'))]
    for text in strings:
        assert '?' not in text and '\ufffd' not in text
        assert re.search(r'[\u0600-\u06ff]',text),repr(text)
        assert text.encode('utf8').decode('utf8')==text
    # ASCII-only expected codepoints cannot share a corrupted shell literal.
    probability=ClauseLanguage.fit(probability_rows('arabic')).templates['probability']
    assert probability.startswith('\u0623\u0642\u062f\u0651\u0631 ')
    assert teacher('arabic')['replacements'][json.loads(Path('results/language_causes_v3_checkpoint.json').read_text(encoding='utf8'))['models']['arabic']['templates']['uncertain']].startswith('\u0644\u064a\u0633 ')
