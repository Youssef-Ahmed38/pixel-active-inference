"""Atomic trusted-local training snapshots; no changes to the learning rule."""
import os
import random
from pathlib import Path

import numpy as np
import torch


def random_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_random(state):
    random.setstate(state['python']);np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def atomic_save(path,payload):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as handle:
        torch.save(payload,handle);handle.flush();os.fsync(handle.fileno())
    os.replace(tmp,path)
