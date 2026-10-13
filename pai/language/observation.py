"""Learn semantic attributes from labeled entity-token sensations.

Uses the existing privileged-state token interface. This does not replace Job A
or claim pixel perception. Labels and color prototypes come from teaching.
"""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np


class LearnedAttributes:
    def __init__(self, prototypes, max_distance=.25):
        self.prototypes = prototypes
        self.max_distance = max_distance

    @classmethod
    def fit(cls, examples, max_distance=.25):
        groups = {"kind": {}, "color": {}}
        for token, labels in examples:
            for attribute, section in (("kind", slice(9, 13)), ("color", slice(13, 16))):
                if attribute in labels:
                    groups[attribute].setdefault(labels[attribute], []).append(np.asarray(token)[section])
        prototypes = {attribute: {label: np.mean(values, axis=0).tolist() for label, values in groups.items()}
                      for attribute, groups in groups.items()}
        return cls(prototypes, max_distance)

    def predict(self, token):
        labels = {}
        for attribute, section in (("kind", slice(9, 13)), ("color", slice(13, 16))):
            values = np.asarray(token, dtype=float)[section]
            if not np.all(np.isfinite(values)):
                continue
            distances = sorted((float(np.linalg.norm(values - prototype)), label)
                               for label, prototype in self.prototypes[attribute].items())
            if (distances and distances[0][0] <= self.max_distance and
                    (len(distances) == 1 or distances[1][0] - distances[0][0] > 1e-6)):
                labels[attribute] = distances[0][1]
        return labels

    def enrich(self, belief, tokens, names):
        if len(tokens) != len(names) or len(set(names)) != len(names):
            raise ValueError("tokens require unique aligned names")
        features = {name: self.predict(token) for name, token in zip(names, tokens)}
        return replace(belief, entities=tuple(replace(entity, features={**entity.features, **features.get(entity.id, {})})
                                              for entity in belief.entities))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema": 1, "prototypes": self.prototypes,
                                    "max_distance": self.max_distance}, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data["schema"] != 1:
            raise ValueError("unsupported attribute checkpoint")
        return cls(data["prototypes"], data["max_distance"])


class GroundedArmAdapter:
    def __init__(self, adapter, attributes):
        self.adapter = adapter
        self.attributes = attributes
        self.domain = adapter.domain

    def observe(self):
        belief = self.adapter.observe()
        return self.attributes.enrich(belief, self.adapter.last_tokens, self.adapter.names)

    def __getattr__(self, name):
        return getattr(self.adapter, name)
