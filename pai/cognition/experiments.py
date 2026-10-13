"""Choose and score short causal probes from predicted binary outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from math import log


def _entropy(probabilities) -> float:
    return -sum(p * log(max(p, 1e-12)) for p in probabilities)


@dataclass(frozen=True)
class BinaryExperiment:
    name: str
    success_given_cause: dict[str, float]
    task_value: float = 0.0
    cost: float = 0.0


class CausalExperimentPlanner:
    def __init__(self, prior: dict[str, float]):
        total = sum(prior.values())
        if not prior or total <= 0:
            raise ValueError("nonempty positive prior required")
        self.prior = {cause: p / total for cause, p in prior.items()}

    def posterior(self, experiment: BinaryExperiment, succeeded: bool) -> dict[str, float]:
        if set(experiment.success_given_cause) != set(self.prior):
            raise ValueError("experiment must predict every cause")
        likelihood = {cause: (p if succeeded else 1 - p)
                      for cause, p in experiment.success_given_cause.items()}
        weights = {cause: self.prior[cause] * max(likelihood[cause], 1e-9)
                   for cause in self.prior}
        normalizer = sum(weights.values())
        return {cause: weight / normalizer for cause, weight in weights.items()}

    def information_gain(self, experiment: BinaryExperiment) -> float:
        p_success = sum(self.prior[cause] * experiment.success_given_cause[cause]
                        for cause in self.prior)
        return (_entropy(self.prior.values())
                - p_success * _entropy(self.posterior(experiment, True).values())
                - (1 - p_success) * _entropy(self.posterior(experiment, False).values()))

    def choose(self, experiments: list[BinaryExperiment]) -> BinaryExperiment:
        if not experiments:
            raise ValueError("at least one experiment required")
        return max(experiments,
                   key=lambda probe: (self.information_gain(probe) + probe.task_value - probe.cost,
                                      probe.name))
