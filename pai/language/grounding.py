"""Cross-situational lexicon and induced templates; no authored word dictionary."""
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re

from pai.cognition.contracts import GoalRelation


@dataclass(frozen=True)
class GroundedFrame:
    intent: str
    predicate: str
    subject_color: str
    subject_kind: str
    target_color: str | None = None
    target_kind: str | None = None
    cause: str | None = None
    truth: str | None = None

    def concepts(self):
        return {(key.rsplit("_", 1)[-1], str(value)) for key, value in asdict(self).items() if value is not None}


def tokenize(text):
    return tuple(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


class GroundedLanguage:
    def __init__(self, lexicon, templates, summary=None):
        self.lexicon = {word: tuple(concept) for word, concept in lexicon.items()}
        self.templates = templates
        self.summary = summary or {}

    @classmethod
    def fit(cls, examples):
        candidates = {}
        for text, frame in examples:
            for word in set(tokenize(text)):
                if word not in candidates:
                    candidates[word] = set(frame.concepts())
                else:
                    candidates[word] &= frame.concepts()
        lexicon = {w: next(iter(values)) for w, values in candidates.items() if len(values) == 1}
        counts = Counter()
        for text, frame in examples:
            fields = {k: str(v) for k, v in asdict(frame).items() if v is not None}
            variants = [((), frozenset())]
            for word in tokenize(text):
                concept = lexicon.get(word)
                next_variants = []
                for pattern, used in variants:
                    matches = [k for k, v in fields.items() if k not in used and concept == (k.rsplit("_", 1)[-1], v)]
                    if matches:
                        next_variants.extend((pattern + (("slot", k),), used | {k}) for k in matches)
                    else:
                        next_variants.append((pattern + (("word", word),), used))
                variants = next_variants
            for pattern, used in variants:
                constants = tuple(sorted((k, v) for k, v in fields.items() if k not in used))
                counts[(pattern, constants)] += 1
        templates = [{"pattern": pattern, "constants": dict(constants), "count": n}
                     for (pattern, constants), n in counts.most_common()]
        return cls(lexicon, templates, {"examples": len(examples), "learned_words": len(lexicon),
                                        "templates": len(templates)})

    def parse(self, text):
        tokens = tokenize(text)
        matches = []
        for template in self.templates:
            if len(template["pattern"]) != len(tokens):
                continue
            fields = dict(template["constants"])
            for token, (kind, value) in zip(tokens, template["pattern"]):
                if kind == "word":
                    if token != value:
                        break
                else:
                    concept = self.lexicon.get(token)
                    if concept is None or concept[0] != value.rsplit("_", 1)[-1]:
                        break
                    fields[value] = concept[1]
            else:
                matches.append((template["count"], GroundedFrame(**fields)))
        if not matches:
            raise ValueError("utterance is outside the learned language")
        best = max(n for n, _ in matches)
        frames = {frame for n, frame in matches if n == best}
        if len(frames) != 1:
            raise ValueError("utterance has ambiguous learned meanings")
        return next(iter(frames))

    def describe(self, frame):
        fields = {k: str(v) for k, v in asdict(frame).items() if v is not None}
        for template in self.templates:
            if any(fields.get(k) != v for k, v in template["constants"].items()):
                continue
            consumed = set(template["constants"])
            words = []
            for kind, value in template["pattern"]:
                if kind == "word":
                    words.append(value)
                else:
                    concept = (value.rsplit("_", 1)[-1], fields.get(value))
                    options = sorted(w for w, learned in self.lexicon.items() if learned == concept)
                    if not options:
                        break
                    words.append(options[0])
                    consumed.add(value)
            else:
                if consumed == set(fields):
                    text = " ".join(words)
                    if self.parse(text) == frame:
                        return text
        raise ValueError("no learned realization for this meaning")

    def goal(self, text, belief):
        frame = self.parse(text)
        if frame.intent != "instruction":
            raise ValueError("an instruction is required")
        def resolve(color, kind):
            choices = [entity.id for entity in belief.entities
                       if entity.confidence > 0 and entity.features.get("color") == color and entity.features.get("kind", entity.kind) == kind]
            if len(choices) != 1:
                raise ValueError("entity description is missing or ambiguous")
            return choices[0]
        subject = resolve(frame.subject_color, frame.subject_kind)
        target = resolve(frame.target_color, frame.target_kind)
        if subject == target:
            raise ValueError("subject and target must be distinct")
        return GoalRelation(frame.predicate, subject, target)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema": 1, "lexicon": self.lexicon, "templates": self.templates,
                                    "summary": self.summary}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data["schema"] != 1:
            raise ValueError("unsupported language checkpoint")
        return cls(data["lexicon"], data["templates"], data["summary"])
