"""Read-only scoring of a completed gate: intervals, confusion and paired costs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from pai.discovery.key_chain_recipe import FAILURE_CAUSES
from scripts.key_chain_v2_eval import wilson


def report(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    rows = data['episodes']
    modes = list(rows[0]['modes'])
    answer = {'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'input': str(path), 'explanations': {}, 'paired': {}, 'examples': []}
    for mode in modes:
        examples = [e for row in rows for e in row['modes'][mode].get('scored_explanations', [])]
        if not examples:
            continue
        k = sum(e['correct'] for e in examples)
        answered = [e for e in examples if e['cause'] != 'unknown']
        matrix = [[sum(e['truth'] == c and e['cause'] == p for e in examples)
                   for p in (*FAILURE_CAUSES, 'unknown')] for c in FAILURE_CAUSES]
        answer['explanations'][mode] = {
            'n': len(examples), 'correct': k, 'accuracy': k / len(examples),
            'wilson95': wilson(k, len(examples)),
            'coverage': len(answered) / len(examples),
            'answered_accuracy': sum(e['correct'] for e in answered) / max(1, len(answered)),
            'row_labels': FAILURE_CAUSES, 'column_labels': [*FAILURE_CAUSES, 'unknown'],
            'confusion': matrix,
            'by_cause': {c: {'n': sum(e['truth'] == c for e in examples),
                             'correct': sum(e['truth'] == c and e['correct'] for e in examples),
                             'wilson95': wilson(sum(e['truth'] == c and e['correct'] for e in examples),
                                                sum(e['truth'] == c for e in examples))}
                         for c in FAILURE_CAUSES}}
    # Resample whole episodes; failed attempts from the same scene are correlated.
    rng = np.random.default_rng(714000)
    indices = rng.integers(0, len(rows), size=(5000, len(rows)))
    for mode in ('cold', 'warm', 'prior_only'):
        counts = np.array([len(r['modes'][mode]['scored_explanations']) for r in rows])
        correct = np.array([sum(e['correct'] for e in r['modes'][mode]['scored_explanations']) for r in rows])
        vals = correct[indices].sum(1) / np.maximum(counts[indices].sum(1), 1)
        answer['explanations'][mode]['episode_bootstrap95'] = list(np.quantile(vals, [.025, .975]))
    solvable = [r for r in rows if r['case'] != 'blocked']
    ix = rng.integers(0, len(solvable), size=(5000, len(solvable)))
    for other in ('cold', 'prior_only', 'scripted'):
        # Show both actual effort and a failure penalty; an early failed script
        # must not count as a faster solution than a successful learned policy.
        delta = np.array([r['modes']['warm']['actions'] - r['modes'][other]['actions'] for r in solvable])
        budget = data['config']['budget']
        cost = lambda r, m: r['modes'][m]['actions'] if r['modes'][m]['success'] else max(budget, r['modes'][m]['actions'])
        penalized = np.array([cost(r, 'warm') - cost(r, other) for r in solvable])
        answer['paired']['warm_minus_' + other] = {
            'mean_actions_all_solvable': float(delta.mean()),
            'episode_bootstrap95': list(np.quantile(delta[ix].mean(1), [.025, .975])),
            'failure_penalized_difference': float(penalized.mean()),
            'failure_penalized_bootstrap95': list(np.quantile(penalized[ix].mean(1), [.025, .975])),
            'warm_faster': int((delta < 0).sum()), 'n': len(solvable)}
    for c in FAILURE_CAUSES:
        candidates = [(r, e) for r in rows for e in r['modes']['warm']['scored_explanations']
                      if e['truth'] == c]
        for good in (True, False):
            pair = next(((r, e) for r, e in candidates if e['correct'] == good), None)
            if pair:
                r, e = pair
                answer['examples'].append({'seed': r['seed'], 'case': r['case'], 'type': r['type'], **e})
    answer['replay'] = {mode: {'actions': sum(r['modes'][mode].get('recipe_actions', 0) for r in rows),
                             'repaired_episodes': sum(bool(r['modes'][mode].get('repair_reason')) for r in rows)}
                        for mode in ('cold', 'warm', 'prior_only')}
    answer['bootstrap'] = {'replicates': 5000, 'seed': 714000, 'unit': 'whole scene'}
    return answer


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    d = report(args.input)
    args.output.write_text(json.dumps(d, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in d.items() if k not in ('examples', 'explanations')}, indent=2))
