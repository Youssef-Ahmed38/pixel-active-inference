import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _seed_everything():
    """Planner and model tests sample randomly; fixed seeds keep them repeatable."""
    torch.manual_seed(0)
    np.random.seed(0)
