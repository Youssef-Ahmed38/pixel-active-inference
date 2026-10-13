"""Teach from actual episode conversations and evaluate disjoint physical layouts."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pai.language import GroundedFrame, GroundedLanguage
from pai.language.teacher import curriculum
from pai.language.conversations import read_episodes, conversation_rows, verify_evidence


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=Path, default=ROOT / "runs/language/arm_episodes.jsonl")
    p.add_argument("--train-seeds", nargs="+", type=int, default=[130, 131])
    p.add_argument("--eval-seeds", nargs="+", type=int, default=[132, 133])
    p.add_argument("--output", type=Path, default=ROOT / "results/language_conversations.json")
    args = p.parse_args()
    if set(args.train_seeds) & set(args.eval_seeds):
        p.error("training and evaluation layouts must be disjoint")
    episodes = read_episodes(args.episodes)
    reports = {}
    for language in ("english", "arabic"):
        rows = conversation_rows(episodes, language)
        teaching = [row for row in rows if row["episode"] in args.train_seeds]
        evaluation = [row for row in rows if row["episode"] in args.eval_seeds]
        if not teaching or not evaluation:
            p.error("episode log must contain training and evaluation layouts")
        curriculum_train, _ = curriculum(language)
        model = GroundedLanguage.fit(curriculum_train + [(row["text"], GroundedFrame(**row["frame"])) for row in teaching])
        model.save(ROOT / f"runs/language/conversations_{language}.json")
        correct = grounded = roundtrips = 0
        failures = []
        for row in evaluation:
            frame = GroundedFrame(**row["frame"])
            grounded += verify_evidence(row, episodes)
            try:
                correct += model.parse(row["text"]) == frame
                roundtrips += model.parse(model.describe(frame)) == frame
            except ValueError as error:
                failures.append({"text": row["text"], "error": str(error)})
        reports[language] = {"scene_teaching_rows": len(teaching), "scene_evaluation_rows": len(evaluation),
            "parsed_exact": correct, "verified_evidence": grounded, "realization_roundtrips": roundtrips,
            "strict_gate_passed": correct == grounded == roundtrips == len(evaluation), "failures": failures[:8]}
        ROOT.joinpath(f"runs/language/conversations_{language}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    report = {"scope": "templates driven by recorded MuJoCo arm observations and intended goals",
              "train_layouts": args.train_seeds, "eval_layouts": args.eval_seeds, "languages": reports,
              "limitations": ["shared templates/vocabulary across layouts; not a novel-grammar test",
                  "observations use privileged tokens", "no unobserved negative or inferred causal claims",
                  "language also receives the separate compositional teaching curriculum"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(reports, ensure_ascii=True, indent=2))
