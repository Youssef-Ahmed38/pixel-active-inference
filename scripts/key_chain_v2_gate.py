"""Checkpointed paired gate with pre-rollout geometric scene validation.

Replacement for the aborted 730000 gate. Policy/likelihood parameters and their
original source hashes remain unchanged; only the evaluator is hardened. Reject
impossible fixture placements before any policy acts, never by agent outcomes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pai.discovery.key_chain_recipe import FailureLikelihood
from pai.maze.env import MazeWorld
from pai.maze.key_chain import KeyChainMaze, _drawer_cell
from scripts.key_chain_v2_eval import (source_hashes, restore_memory, run_learned,
    run_scripted, explain_record, copy_record, summarize)


def run_pair(job, frozen, budget):
    seed, name, case = job['seed'], job['type'], job['case']
    model = FailureLikelihood.from_dict(frozen['failure_model'])
    modes = {}
    for mode in ('cold', 'warm', 'prior_only', 'scripted'):
        try:
            if mode == 'scripted':
                rec = run_scripted(seed, name, case, budget)
            else:
                memory = None if mode == 'cold' else restore_memory(frozen['memory'], mode == 'warm')
                rec, log = run_learned(seed, name, case, budget, memory)
                explain_record(rec, log, model)
            modes[mode] = rec
        except Exception as exc:
            # Never turn an execution failure into a dropped/passing scene.
            modes[mode] = {'seed': seed, 'type': name, 'case': case, 'success': False,
                           'actions': budget, 'actions_to_success': None,
                           'effort_is_failure_penalty': True, 'actual_actions': None,
                           'execution_error': str(exc), 'traceback': traceback.format_exc()}
    modes['no_recipe'] = copy_record(modes['cold'])
    return {**job, 'modes': modes}


def evaluate(args):
    started = time.time()
    frozen = json.loads(args.model.read_text(encoding='utf-8'))
    assert frozen['sources'] == source_hashes(), 'frozen sources changed'
    model_sha = hashlib.sha256(args.model.read_bytes()).hexdigest()
    used = set(frozen['development_seeds']) | set(range(730000, 730048)) | {734000, 734001}
    for p in (ROOT / 'results').glob('key_chain_v2_*.json'):
        d = json.loads(p.read_text(encoding='utf-8'))
        used.update(r['seed'] for r in d.get('episodes', []))
        if 'episode' in d:
            used.add(d['episode']['seed'])
    used.update(s for v in frozen['seed_inventory'].values() if isinstance(v, list) for s in v)
    jobs, rejected = [], []
    seed = args.seed
    cases = ('unlocked', 'latched', 'locked', 'key', 'both', 'blocked')
    while len(jobs) < args.layouts * 6:
        if seed in used:
            raise ValueError(f'previously used seed {seed}')
        maze = MazeWorld(seed=seed, size=15)
        try:
            KeyChainMaze(maze, _drawer_cell(maze))
        except ValueError as exc:
            rejected.append({'seed': seed, 'reason': str(exc)})
            seed += 1
            continue
        i = len(jobs)
        jobs.append({'index': i, 'seed': seed, 'case': cases[i % 6],
                     'type': 'hd_lever_pull_right' if i // 6 < args.layouts // 2 else 'hd_knob_pull_right'})
        seed += 1
    sources = {**source_hashes(), 'scripts/key_chain_v2_gate.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    plan = args.output.with_suffix('.plan.json')
    partial = args.output.with_suffix('.partial.jsonl')
    if plan.exists() or partial.exists():
        raise FileExistsError('existing gate plan/checkpoints')
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    plan.write_text(json.dumps({'config': config, 'sources': sources, 'model_sha256': model_sha,
                                'jobs': jobs, 'rejected_geometry': rejected}, indent=2), encoding='utf-8')
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(run_pair, job, frozen, args.budget): job for job in jobs}
        for future in as_completed(pending):
            row = future.result()
            # Save the actual records before moving on to another pair.
            with partial.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, separators=(',', ':')) + '\n')
                stream.flush()
            rows.append(row)
            print(f"gate {len(rows)}/{len(jobs)} {row['seed']} {row['case']}: " +
                  ', '.join(f"{m} {r['success']}/{r['actions']}" for m, r in row['modes'].items()), flush=True)
    rows.sort(key=lambda r: r['index'])
    assert source_hashes() == frozen['sources'], 'sources changed during evaluation'
    return {'kind': 'fresh_checkpointed_final_gate', 'config': config, 'sources': sources,
            'model_sha256': model_sha, 'rejected_geometry': rejected,
            'prior_aborted_attempt': 'results/key_chain_v2_aborted_gate_730000.json',
            'wall_seconds': time.time() - started,
            'summary': {group: {mode: summarize([r['modes'][mode] for r in rows
                                               if group == 'all' or r['type'] == group])
                                for mode in rows[0]['modes']}
                        for group in ('all', 'hd_lever_pull_right', 'hd_knob_pull_right')},
            'episodes': rows}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--layouts', type=int, default=8)
    p.add_argument('--budget', type=int, default=800)
    p.add_argument('--workers', type=int, choices=(1, 2), default=2)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = evaluate(args)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result['summary'], indent=2))
