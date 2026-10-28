"""Round-3 gates and transparent privileged-token recording."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest


@pytest.mark.parametrize('correct,passed', [(31, False), (32, True)])
def test_gate_a_requires_both_eighty_percent(tmp_path, monkeypatch, correct, passed):
    sys.path.insert(0, str(Path('scripts').resolve()))
    import thinker_hier_round3 as module
    monkeypatch.chdir(tmp_path)
    Path('results').mkdir()
    conditions = ['none', 'push', 'heavier_object', 'slippery_object', 'camera_shift']
    rows = [dict(condition=conditions[i//8], inferred=conditions[i//8] if i<correct else 'unknown',
                 success=i<32) for i in range(40)]
    Path(module.PREFIX+'reproduction_eval.json').write_text(json.dumps({'episodes': rows}))
    monkeypatch.setattr(module, 'register', lambda: {'reproduction_command': ['slice_eval.py']})
    module.score_a()
    result = json.loads(Path(module.PREFIX+'gate_a.json').read_text())
    assert result['passed'] == passed
    rows[0]['success'] = False
    Path(module.PREFIX+'reproduction_eval.json').write_text(json.dumps({'episodes': rows}))
    module.score_a()
    assert not json.loads(Path(module.PREFIX+'gate_a.json').read_text())['passed']


def test_failed_round3_gate_is_binding(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path('scripts').resolve()))
    from thinker_hier_round3_collect import require_gate
    monkeypatch.chdir(tmp_path)
    Path('results').mkdir()
    Path('results/thinker_hier3_gate_b.json').write_text('{"passed":false}')
    with pytest.raises(ValueError, match='prohibited'):
        require_gate('gate_b')


def test_default_token_recorder_preserves_motor_trajectory():
    from pai.agents.slice_agent import run_episode
    from pai.config import load_config
    from pai.envs import TabletopEnv
    from pai.thinker.hier3_tokens import RecordingDefaultPerception, sensory_vector
    from pai.world.model import load_world_model
    cfg = load_config('configs/tabletop.yaml', ['env.render_images=false', 'slice.samples=4',
        'slice.iterations=1', 'slice.horizon=2', 'slice.max_plan_steps=3', 'slice.settle_steps=0'])
    wm = load_world_model(cfg.slice.world_model, 'cpu')
    env = TabletopEnv(cfg.env, disturbances=[])
    try:
        expected = run_episode(env, wm, cfg, 0, 96990, device='cpu', adapt=False)
        recorder = RecordingDefaultPerception(env)
        sensory = []
        actual = run_episode(env, wm, cfg, 0, 96990, device='cpu', adapt=False,
                             perceive=recorder, sensory=sensory)
        assert np.array_equal(actual.tokens, expected.tokens)
        assert np.array_equal(actual.actions, expected.actions)
        assert np.array_equal(np.stack(recorder.tokens), actual.tokens)
        assert len(sensory) == len(actual.actions)
        assert sensory_vector(recorder.tokens[0]).shape == (19,)
        assert np.array_equal(sensory_vector(recorder.tokens[0])[13:16], recorder.tokens[0][1,3:6])
    finally:
        env.close()
