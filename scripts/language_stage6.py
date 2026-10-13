"""Teach compositional grounded English/Arabic and verify unseen combinations."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pai.language import GroundedLanguage
from pai.language.teacher import curriculum
from pai.cognition.contracts import EntityState, StateBelief


def main():
    report = {}
    for language in ("english", "arabic"):
        train, test = curriculum(language)
        model = GroundedLanguage.fit(train)
        model.save(Path("runs/language") / f"{language}.json")
        parsed = realized = goals = instructions = 0
        failures = []
        for text, frame in test:
            try:
                parsed += model.parse(text) == frame
                realized += model.parse(model.describe(frame)) == frame
                if frame.intent == "instruction":
                    instructions += 1
                    belief = StateBelief("arm", 0, (EntityState("object", frame.subject_kind, {"color": frame.subject_color}),
                                                    EntityState("target", frame.target_kind, {"color": frame.target_color})), frozenset())
                    goal = model.goal(text, belief)
                    goals += (goal.predicate, goal.subject, goal.object) == (frame.predicate, "object", "target")
            except ValueError as error:
                if len(failures) < 8:
                    failures.append({"text": text, "error": str(error)})
        report[language] = {"training": model.summary, "heldout_examples": len(test), "parsed_exact": parsed,
                            "realization_roundtrips": realized, "instructions": instructions, "resolved_goals": goals,
                            "strict_suite_passed": parsed == realized == len(test) and goals == instructions,
                            "failures": failures}
    output = {"scope": "template-generated text and symbolic entity observations",
              "evaluation": report, "limitations": ["not fluent conversation or speech recognition",
              "teacher-generated grammar and vocabulary", "physical instruction execution and event-log grounding require their own checks"]}
    Path("results/language_stage6.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
