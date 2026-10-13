from dataclasses import asdict, replace
import copy
import json

from pai.cognition.contracts import CognitiveEpisode, EntityState, StateBelief, GoalRelation
from pai.language.conversations import conversation_rows, verify_evidence


def episode(success=False, observed=True):
    before = StateBelief("arm", 0, (EntityState("a", "block", {"color": "red"}),
                                    EntityState("b", "plate", {"color": "blue"})), frozenset())
    after = replace(before, step=1, relations=frozenset({("on", "a", "b")}) if observed else frozenset())
    record = CognitiveEpisode("arm", 130, GoalRelation("on", "a", "b"), success, 1, beliefs=[before, after])
    return json.loads(json.dumps(asdict(record), default=list))


def test_conversations_have_exact_observed_evidence():
    episodes = [episode()]
    for language in ("english", "arabic"):
        rows = conversation_rows(episodes, language)
        assert len(rows) == 4
        assert all(verify_evidence(row, episodes) for row in rows)
        answer = next(row for row in rows if row["frame"]["intent"] == "answer")
        tampered = copy.deepcopy(answer)
        tampered["frame"]["truth"] = "no"
        assert not verify_evidence(tampered, episodes)
        tampered["frame"]["truth"] = "yes"
        tampered["frame"]["subject_color"] = "green"
        assert not verify_evidence(tampered, episodes)


def test_success_flag_does_not_generate_completed_action_or_cause():
    episodes = [episode(success=True, observed=False)]
    episodes[0]["cause_posterior"] = {"slippery": .99}
    rows = conversation_rows(episodes, "english")
    assert len(rows) == 1 and rows[0]["evidence"] == "intended_goal"
    assert rows[0]["frame"]["cause"] is None
