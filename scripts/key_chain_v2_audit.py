"""Recompute explanations from saved own-evidence traces; no simulator rollouts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pai.discovery.interface import PROBE_KINDS
from pai.discovery.key_chain_recipe import FailureLikelihood, failure_tokens
from scripts.key_chain_v2_eval import source_hashes


def audit(model_path, paths):
    frozen = json.loads(model_path.read_text(encoding='utf-8'))
    assert frozen['sources'] == source_hashes()
    model = FailureLikelihood.from_dict(frozen['failure_model'])
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    counts = {'records': 0, 'predictions': 0, 'scored_explanations': 0, 'no_recipe_pairs': 0}
    inputs = {}
    for path in paths:
        data = json.loads(path.read_text(encoding='utf-8'))
        assert data['model_sha256'] == model_sha
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for row in data['episodes']:
            records = row.get('modes', {'diagnostic': row})
            if 'no_recipe' in records:
                assert records['no_recipe'] == records['cold']
                counts['no_recipe_pairs'] += 1
            for mode, rec in records.items():
                if mode in ('scripted', 'no_recipe') or 'trace' not in rec:
                    continue
                trace = rec['trace']
                for step in trace:
                    assert all(f == 'body:hand' or f.startswith(('part:', 'obj:')) for f in step['before'])
                    assert step['outcome']['notes'] == ''
                    assert all(e['attrs'] == {} for e in step['outcome']['revealed'])
                # Only this explicit allowlist is passed to the predictor.
                own = SimpleNamespace(goal=rec['goal'], actions=[tuple(t['action']) for t in trace],
                                      states=[t['before'] for t in trace],
                                      outcomes=[SimpleNamespace(**t['outcome']) for t in trace])
                predictions = []
                for i, (action, outcome) in enumerate(zip(own.actions, own.outcomes)):
                    if outcome.executed and outcome.stalled and action[0] in (*PROBE_KINDS, 'turn_held'):
                        predictions.append({'step': i, **model.predict(failure_tokens(own, i))})
                assert predictions == rec['explanations']
                by_step = {p['step']: p for p in predictions}
                joined = [{**event, **by_step[event['step']], 'truth': event['cause'],
                           'correct': by_step[event['step']]['cause'] == event['cause']}
                          for event in rec['failure_events'] if event['step'] in by_step]
                assert joined == rec['scored_explanations']
                assert rec['actions'] == len(trace)
                counts['records'] += 1
                counts['predictions'] += len(predictions)
                counts['scored_explanations'] += len(joined)
    return {'passed': True, 'model_sha256': model_sha, 'inputs_sha256': inputs,
            'counts': counts, 'checks': ['frozen inference sources unchanged',
                'same frozen model for all final studies', 'own-evidence feature allowlist',
                'all semantic notes and appearance attributes stripped',
                'all predictions reproduced without truth fields',
                'truth joined only after prediction', 'deterministic no-recipe alias checked'],
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--inputs', nargs='+', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.model, args.inputs)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
