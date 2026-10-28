"""CPU-only development, frozen recipe, paired final gate and event-log scoring.

The simulator answer key lives in ScoredMock only. Agents get EvidenceWorld.
Outputs are append-only: refusing an existing path prevents silent gate reruns.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pai.discovery.agent import _Episode
from pai.discovery.interface import PROBE_KINDS
from pai.discovery.key_chain_recipe import (ChainMemory, EvidenceWorld, FAILURE_CAUSES,
    FailureLikelihood, LearnedChain, RecipeDiscoveryAgent, failure_tokens)
from pai.discovery.mock import MockWorld, Scenario
from pai.discovery.recipes import Recipe


class ScoredMock(MockWorld):
    """Original mock mechanics plus truth-only failed-opening event records.

    A random, permanently obstructed door is an explicit additional mock case.
    It cannot open. No key is forced wrong, no rollout action is forced, and
    role IDs/labels are never included in the policy's outcome or features.
    """
    def __init__(self, scenario, obstructed=False):
        super().__init__(scenario)
        self.obstructed = obstructed
        self.failures = []
        self.current_action = None

    def execute(self, action):
        self.current_action = tuple(action)
        out = super().execute(action)
        if (out.executed and out.stalled and action[0] == "turn_held"
                and self.objs[action[1]]["where"].startswith("in:")
                and not self.objs[action[1]]["matches"]):
            self.failures.append({"step": self.n - 1, "cause": "wrong_key", "action": list(action)})
        return out

    def _move(self, name, fam, released):
        on_leaf = name in ("door", "handle", "thumb", "door_dial", "door_hook", "door_wing")
        opening_attempt = on_leaf and not self.door_open
        # Wrong-direction / ungraspable panel pulls are physical blocking in
        # this discrete mock. A lock has priority over latch when both hold.
        cause = None
        if opening_attempt:
            if fam != self.swing or (name == "door" and fam == "pull"):
                cause = "blocked"
            elif self.thumb == 1 or self.tail == 1:
                matching_rest = any(o["matches"] and o["where"] == f"in:{self.pid.get('cylinder')}"
                                    and o["angle"] == 0 for o in self.objs.values())
                cause = "key_not_turned" if self.tail == 1 and matching_rest and self.thumb == 0 else "locked"
            elif self.latch and not released:
                cause = "latched"
            elif self.obstructed:
                cause = "blocked"
        if opening_attempt and self.obstructed and cause == "blocked" and fam == self.swing:
            stalled = True
        else:
            stalled = super()._move(name, fam, released)
        if opening_attempt and stalled:
            self.failures.append({"step": self.n - 1, "cause": cause or "blocked",
                                  "action": list(self.current_action)})
        return stalled


def wilson(k, n):
    if not n:
        return [None, None]
    z = 1.959963984540054
    p, z2 = k / n, z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return [max(0, mid - half), min(1, mid + half)]


def source_hashes():
    names = ['pai/discovery/key_chain_recipe.py', 'pai/discovery/agent.py',
             'pai/discovery/belief.py', 'pai/discovery/recipes.py', 'pai/discovery/mock.py',
             'pai/maze/key_chain.py', 'scripts/key_chain_v2_eval.py']
    return {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names}


def scan_seeds():
    """Conservative metadata inventory, including nested discovery result dirs."""
    files = []
    for p in (ROOT / 'results').glob('*'):
        if p.name.startswith(('key_chain_', 'discovery_')) and not p.name.startswith('key_chain_v2_'):
            files.extend(p.rglob('*.json') if p.is_dir() else [p] if p.suffix == '.json' else [])
    found = {}
    def walk(x, key=''):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, k)
        elif isinstance(x, list):
            for v in x:
                walk(v, key)
        elif isinstance(x, int) and not isinstance(x, bool) and 'seed' in key.lower():
            seeds.add(x)
    for p in files:
        seeds = set()
        try:
            walk(json.loads(p.read_text(encoding='utf-8')))
            found[str(p.relative_to(ROOT))] = sorted(seeds)
        except (ValueError, OSError):
            found[str(p.relative_to(ROOT))] = 'unreadable'
    return found


def scene(seed, type_name, case):
    state = {'unlocked': 'unlocked', 'latched': 'latch', 'locked': 'deadbolt',
             'key': 'key', 'both': 'both', 'blocked': 'latch'}[case]
    return ScoredMock(Scenario(type_name, state, 'drawer', n_keys=2, seed=seed,
                               p_fail=0.03, pen=True, shared_shapes=True, door_hook=True),
                      obstructed=case == 'blocked')


def run_learned(seed, type_name, case, budget, memory=None, learn=False):
    world = scene(seed, type_name, case)
    agent = RecipeDiscoveryAgent(seed=seed, memory=memory)
    log = agent.run(world, budget)
    if learn and log.reached_goal:
        memory.learn(log, f'development seed {seed}')
    record = record_log(world, log, seed, type_name, case)
    record.update(recipe_actions=agent.replay_actions, repair_reason=agent.repair_reason)
    return record, log


def record_log(world, log, seed, type_name, case):
    record = {'seed': seed, 'type': type_name, 'case': case, 'success': log.reached_goal,
              'actions': log.n_actions, 'actions_to_success': log.n_actions if log.reached_goal else None,
              'time_cost': log.time_cost, 'stop': log.stopped_reason,
              'goal': log.goal, 'belief': log.final_map, 'trace': [
                  {'action': list(a), 'before': st, 'outcome': asdict(o), 'mode': mode}
                  for a, st, o, mode in zip(log.actions, log.states, log.outcomes, log.modes)],
              'events': world.events, 'failure_events': world.failures}
    return record


def serialize_memory(m):
    return {'chains': [r.to_dict() for r in m.chains],
            'priors': [asdict(r) for r in m.priors.recipes]}


def restore_memory(d, sequence=True):
    m = ChainMemory()
    if sequence:
        m.chains = [LearnedChain.from_dict(r) for r in d['chains']]
    for r in d['priors']:
        m.priors.add(Recipe(**r))
    return m


def explain_record(record, log, model):
    # Predictions computed for EVERY executed, stalled probe/inserted turn.
    # The truth-only log is joined afterwards to define the scoring denominator;
    # it is never consulted to select the evidence or generate a prediction.
    predictions = {}
    for i, (a, o) in enumerate(zip(log.actions, log.outcomes)):
        if o.executed and o.stalled and a[0] in (*PROBE_KINDS, 'turn_held'):
            predictions[i] = model.predict(failure_tokens(log, i))
    record['explanations'] = [{'step': i, **p} for i, p in predictions.items()]
    record['scored_explanations'] = [{**event, **predictions[event['step']],
                                     'truth': event['cause'],
                                     'correct': predictions[event['step']]['cause'] == event['cause']}
                                    for event in record['failure_events']
                                    if event['step'] in predictions]


def summarize(rows):
    n = len(rows)
    success = sum(r['success'] for r in rows)
    acts = [r['actions_to_success'] for r in rows if r['success']]
    exp = [e for r in rows for e in r.get('scored_explanations', [])]
    solvable = [r for r in rows if r['case'] != 'blocked']
    solved = sum(r['success'] for r in solvable)
    return {'n': n, 'passed': success, 'success_rate': success / n,
            'wilson95': wilson(success, n),
            'solvable': {'n': len(solvable), 'passed': solved,
                         'wilson95': wilson(solved, len(solvable))},
            'mean_actions_all': sum(r['actions'] for r in rows) / n,
            'mean_actions_to_success': sum(acts) / len(acts) if acts else None,
            'explanations': {'n': len(exp), 'correct': sum(e['correct'] for e in exp),
                'abstained': sum(e['cause'] == 'unknown' for e in exp),
                'by_cause': {c: {'n': sum(e['truth'] == c for e in exp),
                                 'correct': sum(e['truth'] == c and e['correct'] for e in exp),
                                 'abstained': sum(e['truth'] == c and e['cause'] == 'unknown' for e in exp)}
                             for c in FAILURE_CAUSES}}}


def development(args):
    memory = ChainMemory()
    rows, examples = [], []
    # Training/diagnosis development episodes, never used for final estimates.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = []
        for i in range(args.layouts):
            for j, case in enumerate(('unlocked', 'latched', 'locked', 'key', 'both', 'blocked')):
                seed = args.seed + i * 6 + j
                jobs.append(pool.submit(run_learned, seed, 'hd_lever_pull_right', case, args.budget))
        for job in jobs:
            record, log = job.result()
            seed, case = record['seed'], record['case']
            if case == 'both' and log.reached_goal and not memory.chains:
                memory.learn(log, f'development seed {seed}')
            for e in record['failure_events']:
                examples.append((failure_tokens(log, e['step']), e['cause']))
            rows.append(record)
            print(f'dev {seed} {case}: {log.reached_goal} / {log.n_actions}', flush=True)
    if not memory.chains:
        raise RuntimeError('No successful both-lock episode: no chain learned')
    model = FailureLikelihood().fit(examples)
    return {'kind': 'development', 'seed_inventory': scan_seeds(), 'sources': source_hashes(),
            'config': vars_json(args), 'memory': serialize_memory(memory),
            'failure_model': model.to_dict(), 'teaching_examples': len(examples), 'episodes': rows}


def teacher(args):
    """Uniform motor experiments, with no role scoping or forced wrong object.

    Additional development evidence is needed because the discovery policy
    almost always turns a newly inserted object before probing the door.
    """
    rows = []
    for i in range(args.layouts):
        seed = args.seed + i
        case = ('key', 'both', 'latched', 'locked', 'unlocked', 'blocked')[i % 6]
        world = scene(seed, 'hd_lever_pull_right', case)
        body = EvidenceWorld(world)
        ep = _Episode(body)
        rng = np.random.default_rng(seed)
        for _ in range(args.budget):
            acts = body.actions()
            ep.step(acts[int(rng.integers(len(acts)))], 'uniform_experiment')
            if body.goal_reached():
                break
        log = ep.finish('goal' if body.goal_reached() else 'budget')
        rows.append(record_log(world, log, seed, 'hd_lever_pull_right', case))
    return {'kind': 'random_development_teacher', 'sources': source_hashes(),
            'config': vars_json(args), 'episodes': rows}


def freeze(args):
    development_data = json.loads(args.model.read_text(encoding='utf-8'))
    teacher_data = json.loads(args.teacher.read_text(encoding='utf-8'))
    examples = []
    for r in development_data['episodes'] + teacher_data['episodes']:
        trace = r['trace']
        goal = r.get('goal') or scene(r['seed'], r['type'], r['case']).goal()
        log = SimpleNamespace(goal=goal, actions=[tuple(t['action']) for t in trace],
                              states=[t['before'] for t in trace],
                              outcomes=[SimpleNamespace(**t['outcome']) for t in trace])
        for e in r['failure_events']:
            examples.append((failure_tokens(log, e['step']), e['cause']))
    model = FailureLikelihood().fit(examples)
    return {'kind': 'frozen_development_model', 'sources': source_hashes(),
            'config': vars_json(args), 'memory': development_data['memory'],
            'failure_model': model.to_dict(), 'teaching_examples': len(examples),
            'development_seeds': [r['seed'] for r in development_data['episodes'] + teacher_data['episodes']],
            'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (args.model, args.teacher)},
            'seed_inventory': scan_seeds(),
            'episodes': development_data['episodes']}


def run_scripted(seed, type_name, case, budget):
    from pai.maze.env import MazeWorld
    from pai.maze.key_chain import run_key_chain
    world = scene(seed, type_name, case)
    result = run_key_chain(MazeWorld(seed=seed, size=15), world, seed=seed,
                           move_budget=900, fixture_budget=budget)
    # Its per-fixture budget can exceed the new agent's total. Retain actual
    # actions and make this asymmetry explicit rather than hide it.
    return {'seed': seed, 'type': type_name, 'case': case, 'success': result.success,
            'actions': result.fixture_actions,
            'actions_to_success': result.fixture_actions if result.success else None,
            'moves': result.moves, 'stop': result.stage, 'events': world.events,
            'failure_events': world.failures, 'script_reports': result.events}


def evaluate(args):
    frozen = json.loads(args.model.read_text(encoding='utf-8'))
    if frozen['sources'] != source_hashes():
        raise RuntimeError('Sources changed since freezing; create a new development model first')
    dev_seeds = set(frozen.get('development_seeds', [r['seed'] for r in frozen['episodes']]))
    proposed = set(range(args.seed, args.seed + args.layouts * 6))
    used = {s for v in frozen['seed_inventory'].values() if isinstance(v, list) for s in v}
    if proposed & (used | dev_seeds):
        raise ValueError('Evaluation seed overlap')
    model = FailureLikelihood.from_dict(frozen['failure_model'])
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = []
        for i in range(args.layouts):
            for j, case in enumerate(('unlocked', 'latched', 'locked', 'key', 'both', 'blocked')):
                seed = args.seed + i * 6 + j
                type_name = 'hd_lever_pull_right' if i < args.layouts // 2 else 'hd_knob_pull_right'
                jobs.append(pool.submit(evaluate_pair, seed, type_name, case, args.budget,
                                        frozen['memory'], model.to_dict()))
        for job in jobs:
            row = job.result()
            rows.append(row)
            seed, case, paired = row['seed'], row['case'], row['modes']
            print(f"gate {seed} {case}: " + ', '.join(f"{m} {r['success']}/{r['actions']}"
                                                       for m, r in paired.items()), flush=True)
    return {'kind': args.command, 'sources': source_hashes(), 'config': vars_json(args),
            'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
            'summary': {group: {mode: summarize([r['modes'][mode] for r in rows
                                               if group == 'all' or r['type'] == group])
                                for mode in rows[0]['modes']}
                        for group in ('all', 'hd_lever_pull_right', 'hd_knob_pull_right')},
            'episodes': rows}


def evaluate_pair(seed, type_name, case, budget, frozen_memory, failure_model):
    model = FailureLikelihood.from_dict(failure_model)
    paired = {}
    for mode in ('cold', 'warm', 'prior_only', 'no_recipe'):
        if mode == 'no_recipe':
            paired[mode] = copy_record(paired['cold'])
            continue
        memory = None if mode == 'cold' else restore_memory(frozen_memory, mode == 'warm')
        rec, log = run_learned(seed, type_name, case, budget, memory)
        explain_record(rec, log, model)
        paired[mode] = rec
    paired['scripted'] = run_scripted(seed, type_name, case, budget)
    return {'seed': seed, 'type': type_name, 'case': case, 'modes': paired}


def copy_record(record):
    return json.loads(json.dumps(record))


def vars_json(args):
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('develop', 'teacher', 'freeze', 'validate', 'final'))
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--layouts', type=int, default=4)
    parser.add_argument('--budget', type=int, default=800)
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--teacher', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.command in ('validate', 'final', 'freeze') and not args.model:
        parser.error('--model required')
    started = time.time()
    report = {'develop': development, 'teacher': teacher, 'freeze': freeze,
              'validate': evaluate, 'final': evaluate}[args.command](args)
    report['wall_seconds'] = time.time() - started
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report.get('summary', {'teaching_examples': report.get('teaching_examples')})), flush=True)


if __name__ == '__main__':
    main()
