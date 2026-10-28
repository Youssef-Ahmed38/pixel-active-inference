"""Fresh, uniform-experiment diagnostic audit with a frozen cause likelihood.

This separately measures all failure classes; it is NOT the task policy success
gate. No matching object or failed outcome is forced. CPU only.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from pai.discovery.agent import _Episode
from pai.discovery.key_chain_recipe import EvidenceWorld, FailureLikelihood, FAILURE_CAUSES
from scripts.key_chain_v2_eval import scene, record_log, explain_record, source_hashes, wilson


def evaluate(args):
    frozen = json.loads(args.model.read_text(encoding='utf-8'))
    if frozen['sources'] != source_hashes():
        raise ValueError('Frozen inference/collector sources changed')
    seeds = set(range(args.seed, args.seed + args.episodes))
    if seeds.intersection(frozen['development_seeds']):
        raise ValueError('Seed overlap with fitting')
    # Include all earlier v2 episodes (development, validation and task gate),
    # irrespective of their mode, in this diagnostic seed-overlap check.
    inventory = {}
    for p in (ROOT / 'results').glob('key_chain_v2_*.json'):
        d = json.loads(p.read_text(encoding='utf-8'))
        previous = {r['seed'] for r in d.get('episodes', [])}
        if 'episode' in d:
            previous.add(d['episode']['seed'])
        inventory[p.name] = sorted(previous)
        if seeds & previous:
            raise ValueError('Previously used seeds in ' + p.name)
    old = {s for v in frozen['seed_inventory'].values() if isinstance(v, list) for s in v}
    if seeds & old:
        raise ValueError('Previously used Stage 4 seeds')
    model = FailureLikelihood.from_dict(frozen['failure_model'])
    records = []
    for i in range(args.episodes):
        seed = args.seed + i
        case = ('key', 'both', 'latched', 'locked', 'unlocked', 'blocked')[i % 6]
        name = 'hd_lever_pull_right' if i < args.episodes // 2 else 'hd_knob_pull_right'
        world = scene(seed, name, case)
        body = EvidenceWorld(world)
        ep = _Episode(body)
        rng = np.random.default_rng(seed)
        for _ in range(args.budget):
            acts = body.actions()
            ep.step(acts[int(rng.integers(len(acts)))], 'uniform_experiment')
            if body.goal_reached():
                break
        log = ep.finish('goal' if body.goal_reached() else 'budget')
        r = record_log(world, log, seed, name, case)
        explain_record(r, log, model)
        records.append(r)
    events = [e for r in records for e in r['scored_explanations']]
    correct = sum(e['correct'] for e in events)
    counts = np.array([len(r['scored_explanations']) for r in records])
    successes = np.array([sum(e['correct'] for e in r['scored_explanations']) for r in records])
    rng = np.random.default_rng(714001)
    ix = rng.integers(0, len(records), size=(5000, len(records)))
    bootstrap = successes[ix].sum(1) / np.maximum(1, counts[ix].sum(1))
    summary = {'n': len(events), 'correct': correct, 'accuracy': correct / len(events),
               'wilson95': wilson(correct, len(events)),
               'episode_bootstrap95': list(np.quantile(bootstrap, [.025, .975])),
               'abstained': sum(e['cause'] == 'unknown' for e in events),
               'row_labels': FAILURE_CAUSES, 'column_labels': [*FAILURE_CAUSES, 'unknown'],
               'confusion': [[sum(e['truth'] == c and e['cause'] == p for e in events)
                              for p in (*FAILURE_CAUSES, 'unknown')] for c in FAILURE_CAUSES],
               'by_cause': {c: {'n': sum(e['truth'] == c for e in events),
                                'correct': sum(e['truth'] == c and e['correct'] for e in events),
                                'abstained': sum(e['truth'] == c and e['cause'] == 'unknown' for e in events),
                                'wilson95': wilson(sum(e['truth'] == c and e['correct'] for e in events),
                                                   sum(e['truth'] == c for e in events))}
                            for c in FAILURE_CAUSES}}
    return {'kind': 'fresh_random_experiment_diagnostic', 'summary': summary,
            'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
            'sources': {**source_hashes(), 'scripts/key_chain_v2_diagnostic.py':
                        hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            'seed_inventory': inventory,
            'bootstrap': {'seed': 714001, 'replicates': 5000, 'unit': 'whole scene'},
            'episodes': records}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--episodes', type=int, default=120)
    p.add_argument('--budget', type=int, default=800)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = evaluate(args)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result['summary'], indent=2))
