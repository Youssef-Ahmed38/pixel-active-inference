"""Privileged default arm tokens, explicitly given; no new visual decoder."""
import json
from pathlib import Path

import numpy as np

from pai.perception.camera_frame import CameraFramePerception
from pai.thinker.hier_data import load_stream
from pai.world.entities import POS, VEL, FORCE, GRIP


def sensory_vector(tokens, target=1):
    others = [i for i in range(1, len(tokens)) if i != target]
    return np.r_[tokens[0, POS], tokens[0, FORCE], tokens[target, POS],
                 tokens[others, POS].mean(0), tokens[0, GRIP], tokens[target, VEL],
                 tokens[others, VEL].mean(0)].astype(np.float32)


class RecordingDefaultPerception:
    """Transparent recorder of the exact default perception, including settle calls."""
    def __init__(self, env):
        self.default = CameraFramePerception(env)
        self.reset()

    def reset(self):
        self.default.reset()
        self.tokens, self.proprio = [], []

    def __call__(self, obs, prev_ee=None):
        tokens = self.default(obs, prev_ee)
        self.tokens.append(tokens.copy())
        self.proprio.append((obs['q'].copy(), obs['dq'].copy()))
        return tokens


def load_split(split):
    manifests = sorted(Path('results').glob(f'thinker_hier3_{split}_manifest*.json'))
    if not manifests:
        raise ValueError('No round-3 manifests')
    rows = []
    for path in manifests:
        data = json.loads(path.read_text(encoding='utf8'))
        if not data['complete']:
            raise ValueError('Incomplete collection')
        rows.extend(data['episodes'])
    rows.sort(key=lambda r: r['index'])
    if len({r['index'] for r in rows}) != len(rows):
        raise ValueError('Duplicate episode keys')
    return rows, [load_stream(r['path']) for r in rows]


def truth_rows(split):
    rows = []
    for path in Path('results').glob(f'thinker_hier3_{split}_truth*.json'):
        rows.extend(json.loads(path.read_text(encoding='utf8'))['episodes'])
    if len({r['index'] for r in rows}) != len(rows):
        raise ValueError('Duplicate scoring keys')
    return {r['index']: r for r in rows}
