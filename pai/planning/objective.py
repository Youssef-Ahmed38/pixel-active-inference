"""Shared expected-free-energy accounting for body-specific candidate planners.

The domain supplies pragmatic cost and an epistemic information-gain estimate.
The same rule then trades those against action effort and smoothness. Inputs may
be floats, NumPy arrays or Torch tensors; no simulator dependency lives here.
"""

from __future__ import annotations


def expected_free_energy(pragmatic, epistemic, effort=0.0, smoothness=0.0,
                         epistemic_weight: float = 1.0, effort_weight: float = 1.0,
                         smoothness_weight: float = 1.0):
    return (pragmatic - epistemic_weight * epistemic
            + effort_weight * effort + smoothness_weight * smoothness)
