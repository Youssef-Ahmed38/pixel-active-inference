"""Round-2 safety gates and complete sensor capture without changing round 1."""
import json
from types import SimpleNamespace

import numpy as np
import pytest


def study():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path('scripts').resolve()))
    import thinker_hier_round2
    return thinker_hier_round2


def test_round2_uses_normal_motor_settings():
    from pai.config import load_config
    module = study()
    standard = load_config('configs/tabletop.yaml')
    assert module.config().slice.to_dict() == standard.slice.to_dict()


def test_round2_records_every_sensor_call_and_only_own_proprioception(monkeypatch):
    module = study()
    class Observation(dict):
        def __getitem__(self, key):
            assert key in ('ee_pos', 'gripper', 'wrist_force', 'q', 'dq')
            return super().__getitem__(key)
    class Geometry:
        rest_z = np.array([.02, .01])
        def label(self, image): pass
        def masks(self, i): return np.ones((4, 4), bool), None
        def on_plane(self, u, v, z): return np.array([.5, 0, z])
    p = module.RecordingPerception.__new__(module.RecordingPerception)
    p.env = SimpleNamespace(render=lambda c: np.zeros((4, 4, 3), np.uint8), control_dt=.02)
    p.geom = Geometry(); p.camera = 'given'
    p.specs = [SimpleNamespace(kind='block'), SimpleNamespace(kind='plate')]
    p.colours = [[1, 0, 0], [1, 1, 1]]
    p.reset()
    obs = Observation(ee_pos=np.array([.5, 0, .4]), gripper=.08,
                      wrist_force=np.zeros(3), q=np.zeros(7), dq=np.zeros(7))
    for _ in range(6):
        out = p(obs)
        assert np.array_equal(out, p.tokens[-1])
    assert len(p.tokens) == len(p.features) == len(p.proprio) == 6
    p.reset()
    assert not p.tokens and not p.proprio and not p.features


def test_failed_gate_blocks_training_and_final_collection(tmp_path, monkeypatch):
    module = study()
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'results').mkdir()
    monkeypatch.setattr(module, 'register', lambda: {})
    module.save(module.PREFIX + 'gate.json', {'passed': False})
    for split in ('train', 'final'):
        with pytest.raises(ValueError, match='gate failed'):
            module.collect(split, 0, 1, 'cpu')


def test_gate_counts_unrealized_assigned_disturbances_as_errors(tmp_path, monkeypatch):
    module = study()
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'results').mkdir()
    ref = 'results/zero.npz'
    np.savez(ref, residual=np.zeros((25, 12)), sigma=np.ones((25, 12)),
             step=np.array(['carry'] * 25), action=np.zeros((25, 4)), holding=np.ones(25))
    records = [dict(index=i, seed=95000+i%40, assigned_condition=module.CONDITIONS[i//40],
                    disturbances=[], events=[], success=False, reference=ref,
                    steps=25, holding_steps=0, settle_steps=0) for i in range(200)]
    monkeypatch.setattr(module, 'register', lambda: {'data_seeds': {'calibration': [95200]}})
    monkeypatch.setattr(module, 'rows', lambda split: records[:1] if split == 'calibration' else records)
    module.gate()
    result = json.loads((tmp_path / (module.PREFIX+'gate.json')).read_text())
    assert result['accuracy']['k'] == 40
    assert result['accuracy']['n'] == 200 and not result['passed']
    assert result['per_condition']['push']['realized_disturbance_episodes'] == 0
    assert result['per_condition']['push']['accuracy']['k'] == 0
