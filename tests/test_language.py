from dataclasses import replace

import pytest

from pai.cognition.contracts import (CognitiveEpisode, EntityState, GoalRelation,
                                    SkillCommand, SkillOutcome, StateBelief)
from pai.language import GroundedLanguage, LanguageSession
from pai.language.teacher import curriculum


@pytest.fixture(scope="module", params=("english", "arabic"))
def language(request):
    train, test = curriculum(request.param)
    return GroundedLanguage.fit(train), test


class Scene:
    """Symbolic controller fixture, not a physical manipulation benchmark."""
    domain = "scene"

    def __init__(self, frame, stalled=False):
        self.state = StateBelief(self.domain, 0,
            (EntityState("source", frame.subject_kind, {"color": frame.subject_color}),
             EntityState("target", frame.target_kind, {"color": frame.target_color})), frozenset())
        self.stalled = stalled

    def observe(self):
        return self.state

    def available_skills(self):
        return tuple(SkillCommand(p, "source", arguments=("target",)) for p in ("on", "in", "near"))

    def execute(self, command):
        assert command in self.available_skills()
        relations = self.state.relations
        if not self.stalled:
            relations |= {(command.name, command.target, command.arguments[0])}
        self.state = replace(self.state, step=self.state.step + 1, relations=frozenset(relations))
        return SkillOutcome(command, not self.stalled, self.stalled, 1)


def policy(goal):
    return lambda state, adapter: SkillCommand(goal.predicate, goal.subject, arguments=(goal.object,))


def test_heldout_controller_and_observation_narration(language):
    model, test = language
    session = LanguageSession(model)
    for text, frame in test:
        assert model.parse(text) == frame
        assert model.parse(model.describe(frame)) == frame
        if frame.intent != "instruction":
            continue
        scene = Scene(frame)
        record = session.execute(text, scene, policy, budget=2)
        assert record.success and record.steps == 1
        narration = session.narrate(record)
        assert len(narration) == 1
        assert narration[0]["frame"] == replace(frame, intent="description")
        question = model.describe(replace(frame, intent="question"))
        answer = model.parse(session.answer(question, scene.observe()))
        assert answer == replace(frame, intent="answer", truth="yes")


def test_failed_skill_is_not_narrated_as_success(language):
    model, test = language
    text, frame = next(row for row in test if row[1].intent == "instruction")
    scene = Scene(frame, stalled=True)
    session = LanguageSession(model)
    record = session.execute(text, scene, policy, budget=1)
    assert not record.success
    record.cause_posterior = {"slippery": .99}
    narration = session.narrate(record)
    assert len(narration) == 1
    assert narration[0]["frame"].predicate == "stalled"
    assert narration[0]["frame"].cause is None
    record.success = True  # A contradictory success flag must not create a scene fact.
    assert session.narrate(record) == narration


def test_unknown_and_negative_answers(language):
    model, test = language
    _, frame = next(row for row in test if row[1].intent == "question")
    question = model.describe(frame)
    scene = Scene(frame)
    session = LanguageSession(model)
    with pytest.raises(ValueError, match="not been observed"):
        session.answer(question, scene.observe())
    relation = (frame.predicate, "source", "target")
    answer = session.answer(question, scene.observe(), observed_false={relation})
    assert model.parse(answer).truth == "no"
    scene.state = replace(scene.state, relations=frozenset({relation}))
    with pytest.raises(ValueError, match="contradictory"):
        session.answer(question, scene.observe(), observed_false={relation})


def test_checkpoint_unknown_vocabulary_and_ambiguous_entities(language, tmp_path):
    model, test = language
    path = tmp_path / "language.json"
    model.save(path)
    restored = GroundedLanguage.load(path)
    text, frame = next(row for row in test if row[1].intent == "instruction")
    assert restored.parse(text) == frame
    with pytest.raises(ValueError):
        restored.parse(text + " unknownword")
    scene = Scene(frame)
    scene.state = replace(scene.state, entities=scene.state.entities +
                          (replace(scene.state.entities[0], id="duplicate"),))
    with pytest.raises(ValueError, match="ambiguous"):
        LanguageSession(restored).execute(text, scene, policy)


def test_success_without_observation_is_not_narrated(language):
    model, _ = language
    record = CognitiveEpisode("scene", 0, GoalRelation("on", "source", "target"), True, 0)
    assert LanguageSession(model).narrate(record) == []


def test_learned_attributes_use_sensations_and_abstain(tmp_path):
    import numpy as np
    from pai.language import LearnedAttributes
    token = np.zeros(19)
    token[10] = 1
    token[13:16] = [.85, .15, .15]
    decoder = LearnedAttributes.fit([(token, {"kind": "block", "color": "red"})])
    assert decoder.predict(token) == {"kind": "block", "color": "red"}
    unknown = token.copy()
    unknown[13:16] = [0, 1, 0]
    assert decoder.predict(unknown) == {"kind": "block"}
    unknown[:] = np.nan
    assert decoder.predict(unknown) == {}
    path = tmp_path / "attributes.json"
    decoder.save(path)
    assert LearnedAttributes.load(path).predict(token) == decoder.predict(token)
    belief = StateBelief("arm", 0, (EntityState("unrelated_id", "object", {}),), frozenset())
    enriched = decoder.enrich(belief, [token], ["unrelated_id"])
    assert enriched.entities[0].features == {"kind": "block", "color": "red"}
    with pytest.raises(ValueError):
        decoder.enrich(belief, [token], [])


def test_sequence_and_preflight_validation(language):
    model, test = language
    text, frame = next(row for row in test if row[1].intent == "instruction")
    session = LanguageSession(model)
    scene = Scene(frame)
    result = session.execute_sequence([text, text], scene, policy)
    assert result.success and len(result.episodes) == 1
    scene = Scene(frame)
    with pytest.raises(ValueError):
        session.execute_sequence([text, "unknown utterance"], scene, policy)
    assert scene.state.step == 0
    with pytest.raises(ValueError):
        session.execute_sequence([], scene, policy)
