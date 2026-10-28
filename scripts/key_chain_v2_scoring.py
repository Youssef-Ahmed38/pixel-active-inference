"""Expanded failure scoring: include matching-key direction/end-stop stalls.

The original opening-event logger includes wrong-key turns but omitted stalled
matching-key turns. Predictions for these already exist in every saved episode.
Reclassify those omitted *scoring* events as blocked from key_inserted.matches
in the simulator log. Never rerun or refit the frozen agent or predictor.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def evaluate(paths):
    studies = {}
    records = []
    for path in paths:
        data = json.loads(path.read_text(encoding='utf-8'))
        grouped = {}
        for row in data['episodes']:
            for mode, r in row.get('modes', {'diagnostic': row}).items():
                if mode in ('no_recipe', 'scripted') or 'trace' not in r:
                    continue
                scored = list(r['scored_explanations'])
                present = {e['step'] for e in scored}
                predictions = {p['step']: p for p in r['explanations']}
                matches = {e['key']: e['matches'] for e in r['events'] if e['type'] == 'key_inserted'}
                extra = []
                for i, t in enumerate(r['trace']):
                    a, o, s = t['action'], t['outcome'], t['before']
                    if (a[0] == 'turn_held' and o['executed'] and o['stalled'] and i not in present
                            and s.get(f'obj:{a[1]}.where', '').startswith('in:')):
                        assert matches[a[1]] is True
                        p = predictions[i]
                        extra.append({'step': i, 'action': a, **p, 'truth': 'blocked',
                                      'correct': p['cause'] == 'blocked',
                                      'scoring_evidence': 'simulator key_inserted event says matches=True; matching inserted turn stalled'})
                scored += extra
                false_key = sum(e['cause'] == 'wrong_key' and e['truth'] != 'wrong_key' for e in scored)
                key_tp = sum(e['cause'] == 'wrong_key' and e['truth'] == 'wrong_key' for e in scored)
                rec = {'seed': r['seed'], 'mode': mode, 'original_n': len(r['scored_explanations']),
                       'n': len(scored), 'correct': sum(e['correct'] for e in scored),
                       'additional_matching_key_stalls': len(extra), 'wrong_key_tp': key_tp,
                       'wrong_key_false_claims': false_key, 'scored': scored}
                records.append({'study': path.name, **rec})
                grouped.setdefault(mode, []).append(rec)
        summary = {}
        for mode, rows in grouped.items():
            n = sum(r['n'] for r in rows); k = sum(r['correct'] for r in rows)
            tp = sum(r['wrong_key_tp'] for r in rows); fp = sum(r['wrong_key_false_claims'] for r in rows)
            counts = np.array([r['n'] for r in rows]); correct = np.array([r['correct'] for r in rows])
            rng = np.random.default_rng(714002)
            ix = rng.integers(0, len(rows), size=(5000, len(rows)))
            rates = correct[ix].sum(1) / np.maximum(1, counts[ix].sum(1))
            summary[mode] = {'n': n, 'correct': k, 'accuracy': k / n,
                             'episode_bootstrap95': list(np.quantile(rates, [.025, .975])),
                             'additional_matching_key_stalls': sum(r['additional_matching_key_stalls'] for r in rows),
                             'wrong_key_tp': tp, 'wrong_key_false_claims': fp,
                             'wrong_key_precision': tp / (tp + fp) if tp + fp else None}
        studies[path.name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'summary': summary}
    return {'kind': 'expanded_frozen_output_scoring', 'studies': studies, 'records': records,
            'policy_or_likelihood_refit': False, 'new_simulator_rollouts': 0,
            'bootstrap': {'seed': 714002, 'replicates': 5000, 'unit': 'whole scene'},
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = evaluate(args.inputs)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result['studies'], indent=2))
