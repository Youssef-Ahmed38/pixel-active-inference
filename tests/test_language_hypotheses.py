from dataclasses import replace
import math
import pytest

from pai.cognition.contracts import CognitiveEpisode, EntityState, GoalRelation, SkillCommand, SkillOutcome, StateBelief
from pai.language import GroundedLanguage, LanguageSession
from pai.language.teacher import curriculum


def episode():
    belief = StateBelief("arm", 0, (EntityState("source", "block", {"color": "red"}),), frozenset())
    return CognitiveEpisode("arm", 1, GoalRelation("on", "source", "target"), False, 1,
                            beliefs=[belief, replace(belief, step=1)],
                            outcomes=[SkillOutcome(SkillCommand("reach", "source"), True, True, 1)],
                            cause_posterior={"blocked": .9, "heavy": .1})


@pytest.fixture(scope="module", params=["english", "arabic"])
def session(request):
    train, _ = curriculum(request.param, include_hypotheses=True)
    return LanguageSession(GroundedLanguage.fit(train))


def test_uncertain_causes_are_distinct_from_factual_narration(session):
    record = episode()
    facts = session.narrate(record)
    assert all(row["frame"].cause is None for row in facts)
    guesses = session.narrate_hypotheses(record)
    assert len(guesses) == 1
    guess = guesses[0]
    assert session.language.parse(guess["text"]) == guess["frame"]
    assert guess["frame"].intent == "hypothesis" and guess["frame"].cause == "blocked"
    assert guess["confidence"] == .9 and not guess["cause_verified"]
    assert guess["belief_index"] == 1 and guess["outcome_index"] == 0


def test_no_hypotheses_without_stall_observation_or_unambiguous_confidence(session):
    record = episode()
    record.outcomes = [replace(record.outcomes[0], stalled=False)]
    assert session.narrate_hypotheses(record) == []
    record = episode()
    record.beliefs.pop()
    assert session.narrate_hypotheses(record) == []
    record = episode()
    record.cause_posterior = {"blocked": .5, "heavy": .5}
    assert session.narrate_hypotheses(record, minimum_confidence=.1) == []
    record.cause_posterior = {"blocked": .7, "heavy": .3}
    assert session.narrate_hypotheses(record) == []


@pytest.mark.parametrize("posterior", [{"blocked": math.nan}, {"blocked": 1.1}, {"blocked": .3}])
def test_invalid_posteriors_rejected(session, posterior):
    record = episode()
    record.cause_posterior = posterior
    with pytest.raises(ValueError):
        session.narrate_hypotheses(record)


@pytest.mark.parametrize("language", ["english", "arabic"])
def test_heldout_hypothesis_grammar(language):
    train, test = curriculum(language, include_hypotheses=True)
    model = GroundedLanguage.fit(train)
    hypotheses = [(text, frame) for text, frame in test if frame.intent == "hypothesis"]
    assert hypotheses
    for text, frame in hypotheses:
        assert model.parse(text) == frame
        assert model.parse(model.describe(frame)) == frame
