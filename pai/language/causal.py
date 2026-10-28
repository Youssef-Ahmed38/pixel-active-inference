"""Evidence-only causal discourse. No teacher, simulator, or event-log imports.

The semantic schema and confidence policy are given. Surface forms are fitted
from paired teacher clauses, independently for each language. This is supervised
phrase/template induction, not open-ended or cross-situational language learning.
"""
from dataclasses import dataclass
import math
import re


@dataclass
class EpisodeEvidence:
    episode: int
    posterior: dict
    parameters: dict
    adaptations: list
    fallbacks: int
    goal_observed: bool | None
    credits: list

    @classmethod
    def from_record(cls, record, *, goal_observed=None, credits=()):
        # Deliberate allowlist: never read success, events, true_disturbances.
        return cls(record.episode, record.cause_posterior or {}, record.cause_params or {},
                   record.adaptations, record.fallbacks, goal_observed, list(credits))


class ClauseLanguage:
    def __init__(self, templates):
        self.templates = templates

    @classmethod
    def fit(cls, rows):
        templates = {}
        for row in rows:
            text = row['text']
            for name, value in row.get('slots', {}).items():
                text = text.replace(str(value), '{' + name + '}')
            key = row['meaning']
            if key in templates and templates[key] != text:
                raise ValueError('inconsistent paired teaching')
            templates[key] = text
        return cls(templates)

    def say(self, key, **slots):
        return self.templates[key].format(**slots)

    def question(self, text, episode=None):
        normalize = lambda s: re.sub(r'[?؟.!]', '', s).strip().casefold()
        matches = [key[2:].removesuffix(':episode') for key, value in self.templates.items()
                   if key.startswith('q:') and normalize(value.format(episode=episode)) == normalize(text)]
        return matches[0] if len(matches) == 1 else None


class CausalExplainer:
    def __init__(self, language, threshold=.9):
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError('invalid confidence threshold')
        self.language, self.threshold = language, threshold

    def explain(self, evidence):
        p = evidence.posterior
        valid = bool(p) and all(math.isfinite(v) and 0 <= v <= 1 for v in p.values())
        valid = valid and math.isclose(sum(p.values()), 1, abs_tol=1e-6)
        best = max(p, key=p.get) if valid else None
        confidence = p[best] if valid else None
        accepted = (valid and confidence >= self.threshold and
                    sum(v == confidence for v in p.values()) == 1 and
                    best != 'unknown' and 'cause:' + best in self.language.templates)
        cause = best if accepted else None
        say = self.language.say
        clauses = [say('hypothesis', cause=say('cause:' + cause)) if cause else say('uncertain')]
        estimates = {}
        if cause:
            prm = evidence.parameters.get(cause, {})
            for key in ('extra_mass_kg', 'camera_offset_mm', 'onset', 'sink_mm'):
                value = prm.get(key)
                values = value if isinstance(value, list) else [value]
                if value is not None and all(isinstance(v, (float, int)) and math.isfinite(v) for v in values):
                    estimates[key] = value
                    clauses.append(say('estimate:' + key, value=', '.join(f'{v:g}' for v in values)))
        # Actions are reported as recorded, independently of the final verdict.
        actions = []
        for action in evidence.adaptations:
            key = 'action:' + action['action']
            if key in self.language.templates and key not in actions:
                actions.append(key)
                clauses.append(say(key))
        if evidence.fallbacks:
            clauses.append(say('fallback', value=evidence.fallbacks))
        clauses.append(say('goal:unknown' if evidence.goal_observed is None else
                           'goal:yes' if evidence.goal_observed else 'goal:no'))
        return dict(text=' '.join(clauses), cause=cause, confidence=confidence,
                    abstained=not accepted, estimates=estimates, actions=actions,
                    goal_observed=evidence.goal_observed, evidence_episode=evidence.episode)

    def answer(self, question, evidence):
        intent = self.language.question(question, evidence.episode)
        if intent == 'why':
            return self.explain(evidence)
        if intent == 'drop':
            # A fallback is a loss-of-hold belief, not proof of a cause of a drop.
            return dict(text=self.language.say('drop:unknown'), abstained=True, intent=intent)
        if intent and intent.startswith('cf:'):
            name = intent[3:]
            matches = [r for r in evidence.credits if r['name'] == name and r['valid']
                       and all(math.isfinite(r[k]) for k in ('goal_cost', 'baseline_cost'))]
            if len(matches) == 1:
                r = matches[0]
                return dict(text=self.language.say('cf:necessary' if r['necessary'] else 'cf:unnecessary'),
                            abstained=False, intent=intent, credit=r)
        return dict(text=self.language.say('cf:unknown'), abstained=True, intent=intent)
