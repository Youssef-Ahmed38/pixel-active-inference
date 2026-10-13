"""Teach uncertain causal language separately and audit recorded stall guesses."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pai.cognition.contracts import CognitiveEpisode, GoalRelation, SkillCommand, SkillOutcome
from pai.language import GroundedLanguage, LanguageSession
from pai.language.conversations import read_episodes, belief_from_record
from pai.language.teacher import curriculum


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=Path, default=ROOT / "runs/language/conversation_episodes.jsonl")
    p.add_argument("--artifacts", type=Path, default=ROOT / "runs/language/hypotheses")
    p.add_argument("--output", type=Path, default=ROOT / "results/language_hypotheses.json")
    args = p.parse_args()
    episodes = []
    for data in read_episodes(args.episodes):
        outcomes = [SkillOutcome(command=SkillCommand(**outcome["command"]),
                                **{k: v for k, v in outcome.items() if k != "command"})
                    for outcome in data.get("outcomes", [])]
        episodes.append(CognitiveEpisode(data["domain"], data["episode"], GoalRelation(**data["goal"]),
            data["success"], data["steps"], beliefs=[belief_from_record(b) for b in data["beliefs"]],
            outcomes=outcomes, cause_posterior=data.get("cause_posterior", {})))
    reports = {}
    args.artifacts.mkdir(parents=True, exist_ok=True)
    for language in ("english", "arabic"):
        train, test = curriculum(language, include_hypotheses=True)
        model = GroundedLanguage.fit(train)
        model.save(args.artifacts / f"{language}.json")
        parsed = roundtrips = hypotheses = hypothesis_parses = 0
        for text, frame in test:
            exact = model.parse(text) == frame
            parsed += exact
            roundtrips += model.parse(model.describe(frame)) == frame
            if frame.intent == "hypothesis":
                hypotheses += 1
                hypothesis_parses += exact
        rows = []
        for index, episode in enumerate(episodes):
            for row in LanguageSession(model).narrate_hypotheses(episode):
                rows.append({**row, "frame": asdict(row["frame"]), "source_episode_index": index,
                             "domain": episode.domain, "episode": episode.episode})
        (args.artifacts / f"{language}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        reports[language] = {"heldout_examples": len(test), "parsed_exact": parsed,
                             "realization_roundtrips": roundtrips, "heldout_hypotheses": hypotheses,
                             "hypotheses_parsed_exact": hypothesis_parses,
                             "grammar_gate_passed": parsed == roundtrips == len(test),
                             "recorded_episode_count": len(episodes), "recorded_hypotheses": len(rows),
                             "verified_causes": 0}
    report = {"scope": "uncertain cause language tied to observed stalls",
              "languages": reports,
              "limitations": ["an episode posterior is not verified causal evidence or a calibrated probability",
                              "no hypothesis is generated without a normalized posterior and corresponding stall observation",
                              "heldout grammar is teacher generated; physical causal verification remains unfinished"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
