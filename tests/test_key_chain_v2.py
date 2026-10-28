"""Fast recipe, evidence-boundary and learned diagnosis tests (CPU only)."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from pai.discovery.agent import _Episode
from pai.discovery.key_chain_recipe import (ChainMemory, EvidenceWorld, FAILURE_CAUSES,
    FailureLikelihood, LearnedChain, RecipeDiscoveryAgent, failure_tokens)
from pai.discovery.mock import MockWorld, Scenario
from scripts.key_chain_v2_eval import ScoredMock, explain_record, summarize, wilson


def demonstrated(seed=710900):
    # Teacher uses simulator roles to construct a unit-test demonstration. The
    # learner receives ONLY the resulting observation/action EpisodeLog.
    sc = Scenario('hd_lever_pull_right', 'both', 'drawer', n_keys=1,
                  seed=seed, p_fail=0, pen=False, dial=False, cabinet=False)
    world = MockWorld(sc)
    p, key = world.pid, world.truth()['matching_key']
    ep = _Episode(EvidenceWorld(world))
    for a in [('pull', p['drawer']), ('push', p['drawer']), ('pull', p['drawer']),
              ('pick', key), ('insert', key, p['cylinder']),
              ('turn_held', key, world.unlock_sign), ('release', key),
              ('turn', p['thumb'], -1), ('turn_pull', p['handle'], -1)]:
        ep.step(a, 'demonstration')
    log = ep.finish('goal')
    log.final_map = {'probe': log.actions[-1],
                     'literals': {f'part:{p["thumb"]}.angle': 'rest',
                                  f'obj:{key}.angle': 'pos' if world.unlock_sign == 1 else 'neg'},
                     'p': .8}
    return sc, log


def test_recipe_learns_order_preconditions_and_discards_exploration_loops():
    sc, log = demonstrated()
    frozen = copy.deepcopy(log)
    memory = ChainMemory()
    r = memory.learn(log)
    assert r is not None
    assert len(r.steps) == 7  # drawer open/close/open loop is compacted
    kinds = [s.action[0] for s in r.steps]
    assert kinds.index('pull') < kinds.index('pick') < kinds.index('insert') < kinds.index('turn_held')
    assert all(e.startswith('r') for e in r.entities)
    assert all('shape' not in d and 'size' not in d for d in r.entities.values())
    assert LearnedChain.from_dict(r.to_dict()).to_dict() == r.to_dict()
    assert [s.action for s in log.outcomes] == [s.action for s in frozen.outcomes]
    ag = RecipeDiscoveryAgent(seed=0, memory=memory)
    new = ag.run(MockWorld(sc), 10)
    assert new.reached_goal and new.n_actions == 7
    assert ag.replay_actions == 7 and not ag.disabled


def test_recipe_rejection_returns_to_discovery_without_key_specific_rules():
    sc, log = demonstrated()
    memory = ChainMemory(); memory.learn(log)
    # A corrupted demonstrated effect is generic model surprise, not a rule
    # about two turns stalling or an incompatible object's shape.
    memory.chains[0].steps[0].effects['body:hand'] = 'impossible'
    ag = RecipeDiscoveryAgent(seed=0, memory=memory)
    ag.run(MockWorld(sc), 2)
    assert ag.disabled and ag.replay_actions == 1
    assert 'contradicted' in ag.repair_reason
    assert np.ptp(ag.belief.log_prior[ag.belief.h_size == 0]) == 0


def test_evidence_boundary_removes_semantics_and_never_reads_truth():
    class Guard(MockWorld):
        def truth(self):
            raise AssertionError('truth accessed')
    world = Guard(Scenario('hd_lever_pull_right', 'latch', 'drawer', seed=710901, p_fail=0))
    body = EvidenceWorld(world)
    assert not hasattr(body, 'pid') and not hasattr(body, 'truth') and not hasattr(body, 'events')
    assert all(e.attrs == {} for e in body.entities())
    log = RecipeDiscoveryAgent(seed=0).run(world, 40)
    assert log.n_actions > 0
    assert all(o.notes == '' for o in log.outcomes)


def test_repair_cycle_uses_an_untried_action_context():
    ag = RecipeDiscoveryAgent(seed=0)
    ag.disabled = True
    state = {'part:p1.angle': 'rest', 'body:hand': 'empty'}
    ep = SimpleNamespace(state=state, log=SimpleNamespace(states=[dict(state)] * 3))
    acts = [('turn', 'p1', 1), ('push', 'p2')]
    ag.model = SimpleNamespace(untested=lambda a, st: a == acts[1])
    assert ag._exploit(ep, acts) == (acts[1], 'repair_experiment')


def test_situated_body_reads_passage_guard_and_reuses_one_map():
    from pai.discovery.key_chain_body import SituatedChainBody
    from pai.maze.env import MazeWorld
    fixture = MockWorld(Scenario('hd_lever_pull_right', 'latch', 'drawer', seed=710904, p_fail=0))
    maze = MazeWorld(seed=710904, size=15)
    body = SituatedChainBody(maze, fixture, seed=0)
    assert not body.pass_goal() and not body.entered_exit
    memory = body.navigator.memory
    out = body.execute(('turn_pull', fixture.pid['handle'], -1))
    assert not out.stalled and fixture.goal_reached()
    assert body.pass_goal() and maze.position == maze.goal
    assert body.navigator.memory is memory
    assert not hasattr(body, 'pid') and not hasattr(body, 'door_open')


def test_situated_scene_rejects_unreachable_workspace_instead_of_crossing_locked_exit():
    from pai.discovery.key_chain_body import SituatedChainBody
    from pai.maze.env import MazeWorld
    fixture = MockWorld(Scenario('hd_lever_pull_right', 'key', 'drawer', seed=734001, p_fail=0))
    with pytest.raises(ValueError, match='no reachable remote workspace'):
        SituatedChainBody(MazeWorld(seed=734001, size=15), fixture)


def test_gate_keeps_learned_results_when_legacy_baseline_raises(monkeypatch):
    from scripts import key_chain_v2_gate as gate
    def learned(seed, name, case, budget, memory):
        return ({'seed': seed, 'type': name, 'case': case, 'success': True,
                 'actions': 1, 'actions_to_success': 1}, None)
    def broken(*args):
        raise ValueError('invalid legacy placement')
    monkeypatch.setattr(gate, 'run_learned', learned)
    monkeypatch.setattr(gate, 'explain_record', lambda *args: None)
    monkeypatch.setattr(gate, 'run_scripted', broken)
    frozen = {'failure_model': FailureLikelihood().to_dict(),
              'memory': {'chains': [], 'priors': []}}
    row = gate.run_pair({'seed': 710905, 'type': 'opaque', 'case': 'key'}, frozen, 10)
    assert all(row['modes'][m]['success'] for m in ('cold', 'warm', 'prior_only'))
    assert not row['modes']['scripted']['success']
    assert row['modes']['scripted']['execution_error'] == 'invalid legacy placement'
    assert row['modes']['scripted']['effort_is_failure_penalty']
    assert row['modes']['no_recipe'] == row['modes']['cold']


def test_failure_likelihood_is_learned_and_roundtrips():
    rows = [(['kind=' + str(i), 'value=' + str(i)], c)
            for i, c in enumerate(FAILURE_CAUSES) for _ in range(20)]
    m = FailureLikelihood().fit(rows)
    restored = FailureLikelihood.from_dict(m.to_dict())
    shuffled = FailureLikelihood().fit([(tokens, FAILURE_CAUSES[(i // 20 + 1) % 5])
                                       for i, (tokens, _) in enumerate(rows)])
    for i, c in enumerate(FAILURE_CAUSES):
        tokens = ['kind=' + str(i), 'value=' + str(i)]
        p = m.predict(tokens)
        assert p['cause'] == c and restored.predict(tokens) == p
        assert np.isclose(sum(p['posterior'].values()), 1)
        assert shuffled.predict(tokens)['cause'] != c
    assert m.predict(['kind=unseen', 'value=unseen'])['cause'] == 'unknown'


def test_event_labels_and_prediction_denominator_are_separate():
    world = ScoredMock(Scenario('hd_lever_pull_right', 'key', 'drawer',
                                n_keys=1, seed=710902, p_fail=0))
    ep = _Episode(EvidenceWorld(world)); p = world.pid
    ep.step(('pull', p['handle']), 'test')
    ep.step(('pull', p['drawer']), 'test')
    key = world.truth()['matching_key']
    ep.step(('pick', key), 'test')
    ep.step(('insert', key, p['cylinder']), 'test')
    ep.step(('release', key), 'test')
    ep.step(('turn_pull', p['handle'], -1), 'test')
    log = ep.finish('budget')
    assert world.failures[0]['cause'] == 'locked'
    assert world.failures[-1]['cause'] == 'key_not_turned'
    model = FailureLikelihood().fit([(failure_tokens(log, 0), 'locked')])
    rec = {'failure_events': world.failures}
    explain_record(rec, log, model)
    before = copy.deepcopy(rec['explanations'])
    rec['failure_events'] = []
    explain_record(rec, log, model)
    assert rec['explanations'] == before  # truth cannot change words/posteriors
    assert rec['scored_explanations'] == []


def test_blocked_mock_cannot_be_opened_by_policy_and_metrics_include_failures():
    world = ScoredMock(Scenario('hd_lever_pull_right', 'latch', 'drawer', seed=710903, p_fail=0), True)
    out = world.execute(('turn_pull', world.pid['handle'], -1))
    assert out.stalled and not world.goal_reached()
    assert world.failures[-1]['cause'] == 'blocked'
    r = summarize([{'success': False, 'case': 'key', 'actions': 20, 'actions_to_success': None},
                   {'success': False, 'case': 'blocked', 'actions': 20, 'actions_to_success': None}])
    assert r['passed'] == 0 and r['solvable']['n'] == 1
    assert r['mean_actions_to_success'] is None
    assert np.isclose(wilson(0, 10)[0], 0) and np.isclose(wilson(10, 10)[1], 1)
