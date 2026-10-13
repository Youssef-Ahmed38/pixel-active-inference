"""Grounded text interface to the existing cognitive controller and observations.

Absence from a partial belief does not establish falsity. Causal posterior scores
are hypotheses, not observed explanations, and are never narrated as facts here.
"""
from dataclasses import replace
import math

from pai.cognition.controller import CognitiveController
from pai.language.grounding import GroundedFrame


class LanguageSession:
    def __init__(self, language, controller=None):
        self.language = language
        self.controller = controller or CognitiveController()

    @staticmethod
    def resolve(frame, belief):
        def entity(color, kind):
            matches = [e for e in belief.entities if e.confidence > 0 and
                       e.features.get("color") == color and
                       e.features.get("kind", e.kind) == kind]
            if len(matches) != 1:
                raise ValueError("entity description is missing or ambiguous")
            return matches[0].id
        return (entity(frame.subject_color, frame.subject_kind),
                entity(frame.target_color, frame.target_kind) if frame.target_kind else None)

    def answer(self, text, belief, *, observed_false=frozenset()):
        frame = self.language.parse(text)
        if frame.intent != "question":
            raise ValueError("a question is required")
        subject, target = self.resolve(frame, belief)
        relation = (frame.predicate, subject, target)
        if relation in belief.relations and relation in observed_false:
            raise ValueError("contradictory observations")
        if relation in belief.relations:
            truth = "yes"
        elif relation in observed_false:
            truth = "no"
        else:
            raise ValueError("the relation has not been observed")
        return self.language.describe(replace(frame, intent="answer", truth=truth))

    def execute(self, text, adapter, policy_factory, *, budget=100, episode=0):
        goal = self.language.goal(text, adapter.observe())
        return self.controller.run(adapter, goal, policy_factory(goal), budget, episode)

    def execute_sequence(self, instructions, adapter, policy_factory, *, budget_per_goal=100,
                         episode=0, max_passes=2, repair_factory=None):
        if not instructions:
            raise ValueError("at least one instruction is required")
        belief = adapter.observe()
        goals = [self.language.goal(text, belief) for text in instructions]
        return self.controller.run_sequence(adapter, goals, policy_factory, budget_per_goal,
                                            episode, max_passes, repair_factory)

    @staticmethod
    def frame(belief, predicate, subject, target=None):
        entities = {e.id: e for e in belief.entities}
        source = entities[subject]
        destination = entities[target] if target is not None else None
        return GroundedFrame("description", predicate, source.features["color"],
                             source.features.get("kind", source.kind),
                             None if destination is None else destination.features["color"],
                             None if destination is None else destination.features.get("kind", destination.kind))

    def narrate(self, episode):
        """Describe observed relation gains and stalls, each tied to a log index.

        Success flags alone cannot manufacture an observed relation. Unsupported
        vocabulary is skipped; no simulator access or cause inference is used.
        """
        records = []
        for index, (before, after) in enumerate(zip(episode.beliefs, episode.beliefs[1:])):
            for predicate, subject, target in sorted(after.relations - before.relations, key=str):
                try:
                    frame = self.frame(after, predicate, subject, target)
                    text = self.language.describe(frame)
                except (KeyError, ValueError):
                    continue
                records.append({"text": text, "frame": frame, "belief_index": index + 1,
                                "evidence": "observed_relation_gain"})
        for index, outcome in enumerate(episode.outcomes):
            if not outcome.stalled or not outcome.command.target or not episode.beliefs:
                continue
            belief = episode.beliefs[min(index + 1, len(episode.beliefs) - 1)]
            try:
                frame = self.frame(belief, "stalled", outcome.command.target)
                text = self.language.describe(frame)
            except (KeyError, ValueError):
                continue
            records.append({"text": text, "frame": frame, "outcome_index": index,
                            "evidence": "observed_skill_stall"})
        return records

    def narrate_hypotheses(self, episode, *, minimum_confidence=0.8):
        """Attach an explicitly uncertain episode-level guess to observed stalls.

        The posterior is not a verified cause or calibrated probability. Every
        returned record retains its stall index and hypothesis confidence.
        """
        if not math.isfinite(minimum_confidence) or not 0 <= minimum_confidence <= 1:
            raise ValueError("confidence threshold must be between zero and one")
        posterior = episode.cause_posterior
        if not posterior:
            return []
        if (any(not math.isfinite(p) or not 0 <= p <= 1 for p in posterior.values())
                or not math.isclose(sum(posterior.values()), 1.0, abs_tol=1e-6)):
            raise ValueError("cause posterior must contain finite normalized probabilities")
        confidence = max(posterior.values())
        causes = [cause for cause, p in posterior.items() if p == confidence]
        if confidence < minimum_confidence or len(causes) != 1:
            return []
        records = []
        for index, outcome in enumerate(episode.outcomes):
            if not outcome.stalled or not outcome.command.target or index + 1 >= len(episode.beliefs):
                continue
            # Require the corresponding observation, never borrow a later one.
            belief = episode.beliefs[index + 1]
            try:
                frame = replace(self.frame(belief, "stalled", outcome.command.target),
                                intent="hypothesis", cause=causes[0])
                text = self.language.describe(frame)
            except (KeyError, ValueError):
                continue
            records.append({"text": text, "frame": frame, "outcome_index": index,
                            "belief_index": index + 1, "confidence": confidence,
                            "evidence": "observed_stall_with_episode_cause_hypothesis",
                            "cause_verified": False})
        return records
