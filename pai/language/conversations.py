"""Teacher-owned conversations grounded in serialized cognitive observations.

Each statement records its supporting belief or outcome index. Goal instructions
describe intended goals; they are never mislabeled as completed actions. Missing
relations and posterior cause guesses do not become negative or causal facts.
"""
from dataclasses import asdict, replace
import json
from pathlib import Path

from pai.cognition.contracts import EntityState, StateBelief
from pai.language.session import LanguageSession
from pai.language.teacher import utterance


def read_episodes(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def belief_from_record(record):
    return StateBelief(record["domain"], record["step"], tuple(EntityState(**e) for e in record["entities"]),
                       frozenset(tuple(relation) for relation in record["relations"]), record.get("uncertainty", 0))


def conversation_rows(episodes, language):
    rows = []
    for episode_index, episode in enumerate(episodes):
        beliefs = [belief_from_record(record) for record in episode["beliefs"]]
        seen = set()
        def add(frame, evidence, index, relation=None):
            try:
                text = utterance(frame, language)
            except KeyError:
                return
            key = (frame, evidence, relation)
            if key in seen:
                return
            seen.add(key)
            rows.append({"text": text, "frame": asdict(frame), "domain": episode["domain"],
                "episode": episode["episode"], "source_episode_index": episode_index,
                "evidence": evidence, "index": index, "relation": relation})
        if beliefs:
            goal = episode["goal"]
            try:
                frame = LanguageSession.frame(beliefs[0], goal["predicate"], goal["subject"], goal["object"])
                add(replace(frame, intent="instruction"), "intended_goal", 0)
            except KeyError:
                pass
        for index, belief in enumerate(beliefs):
            for relation in sorted(belief.relations, key=str):
                try:
                    frame = LanguageSession.frame(belief, *relation)
                except KeyError:
                    continue
                add(frame, "observed_relation", index, relation)
                if frame.target_kind is not None:
                    add(replace(frame, intent="question"), "observed_relation", index, relation)
                    add(replace(frame, intent="answer", truth="yes"), "observed_relation", index, relation)
        for index, outcome in enumerate(episode.get("outcomes", ())):
            subject = outcome["command"].get("target")
            if outcome["stalled"] and subject and beliefs:
                try:
                    frame = LanguageSession.frame(beliefs[min(index + 1, len(beliefs) - 1)], "stalled", subject)
                    add(frame, "observed_stall", index)
                except KeyError:
                    pass
    return rows


def verify_evidence(row, episodes):
    episode = episodes[row["source_episode_index"]]
    if episode["episode"] != row["episode"] or episode["domain"] != row["domain"]:
        return False
    if row["evidence"] == "observed_relation":
        belief = belief_from_record(episode["beliefs"][row["index"]])
        relation = tuple(row["relation"])
        if relation not in belief.relations or row["frame"]["intent"] not in ("description", "question", "answer"):
            return False
        expected = replace(LanguageSession.frame(belief, *relation), intent=row["frame"]["intent"],
                           truth="yes" if row["frame"]["intent"] == "answer" else None)
        return asdict(expected) == row["frame"]
    if row["evidence"] == "intended_goal":
        goal = episode["goal"]
        relation = (goal["predicate"], goal["subject"], goal["object"])
        belief = belief_from_record(episode["beliefs"][0])
        expected = replace(LanguageSession.frame(belief, *relation), intent="instruction")
        return asdict(expected) == row["frame"]
    if row["evidence"] == "observed_stall":
        outcome = episode["outcomes"][row["index"]]
        belief = belief_from_record(episode["beliefs"][min(row["index"] + 1, len(episode["beliefs"]) - 1)])
        expected = LanguageSession.frame(belief, "stalled", outcome["command"]["target"])
        return bool(outcome["stalled"] and asdict(expected) == row["frame"])
    return False
