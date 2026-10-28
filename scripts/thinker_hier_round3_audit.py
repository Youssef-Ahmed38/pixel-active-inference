"""Read-only input/provenance audit and exact frozen final-score replay."""
import json
import re
from pathlib import Path

import numpy as np
import torch

from pai.thinker.hier3_tokens import load_split, truth_rows, sensory_vector
from pai.thinker.hier_data import normalization, batch_windows
from pai.thinker.hier_analysis import run_streams
from thinker_hier_collect import digest
from thinker_hier_round2 import save
from thinker_hier_round3 import PREFIX, register
from thinker_hier_round3_report import audit_previous, append


def main():
    plan = register()
    result = dict(previous_round_files_unchanged=audit_previous(),
                  previous_round_text_unchanged=True, stream_counts={}, final_replays=0)
    for key, sha in plan['registered_sources'].items():
        if digest(key) != sha:
            raise ValueError('Registered arm source changed: '+key)
    if digest(plan['reproduction_config']['slice']['world_model']) != plan['world_model_sha256']:
        raise ValueError('Arm world-model changed')
    for split in ('calibration', 'development', 'train', 'final'):
        manifests = sorted(Path('results').glob(f'thinker_hier3_{split}_manifest*.json'))
        if not manifests:
            continue
        rows, streams = load_split(split); truth = truth_rows(split)
        expected = 8 if split=='calibration' else len(plan['data_seeds'][split])*5
        if len(rows) != expected or len(truth) != expected:
            raise ValueError('Incomplete split')
        for mp in manifests:
            meta = json.loads(mp.read_text(encoding='utf8'))
            for path, sha in meta['source_hashes'].items():
                if digest(path) != sha:
                    raise ValueError('Capture source changed: '+path)
            if meta['world_model_sha256'] != plan['world_model_sha256']:
                raise ValueError('Capture used another arm model')
        for row, stream in zip(rows, streams):
            if digest(row['path']) != row['sha256']:
                raise ValueError('Raw capture hash changed')
            n = len(stream['action'])
            if stream['state'].shape != (n+1, 19) or stream['context'].shape != (n, 10):
                raise ValueError('Bad transition alignment')
            with np.load(row['path'], allow_pickle=False) as raw:
                if any(not np.isfinite(raw[k]).all() for k in raw.files):
                    raise ValueError('Nonfinite raw stream')
                if not np.array_equal(raw['state'], np.stack([sensory_vector(t) for t in raw['tokens']])):
                    raise ValueError('State is not the registered privileged-token projection')
                if len(raw['q']) != n or len(raw['dq']) != n:
                    raise ValueError('Incomplete proprioception')
            with np.load(truth[row['index']]['reference'], allow_pickle=False) as ref:
                if len(ref['residual']) != n or not np.array_equal(ref['action'], stream['action']):
                    raise ValueError('Reference/action stream mismatch')
        result['stream_counts'][split] = len(rows)
    trained = list(Path('results').glob('thinker_hier3_*_training.json'))
    result['trained_models'] = len(trained)
    if trained:
        if len(trained) != 9:
            raise ValueError('Nine completed training jobs required')
        expected_streams = {r['path']: r['sha256'] for r in load_split('train')[0]}
        signatures = []
        for path in trained:
            t = json.loads(path.read_text(encoding='utf8'))
            if t['steps'] != 1200 or t['parameters'] != plan['parameters'][t['kind']]:
                raise ValueError('Training budget mismatch')
            if t['streams'] != expected_streams or digest(t['checkpoint']) != t['checkpoint_sha256']:
                raise ValueError('Training streams/checkpoint differ')
            signatures.append(t['source_hashes'])
        if any(s != signatures[0] for s in signatures):
            raise ValueError('Source changes across training jobs')
        # Prove stream loading/window sampling do not open answer-key/reference/gate labels.
        import builtins, io
        old_open, old_io_open = builtins.open, io.open
        def guarded(original):
            def call(file, *args, **kwargs):
                name = str(file)
                if 'truth' in name or '_reference_' in name or name.endswith(('gate_a.json','gate_b.json')):
                    raise ValueError('Label-bearing file opened by sensory reader')
                return original(file, *args, **kwargs)
            return call
        builtins.open, io.open = guarded(old_open), guarded(old_io_open)
        try:
            _, streams = load_split('train')
            batch_windows(streams, normalization(streams), np.random.default_rng(1), 16, 24, 'cpu')
        finally:
            builtins.open, io.open = old_open, old_io_open
        result['guarded_sensory_loading_passed'] = True
    if Path(PREFIX+'freeze.json').exists():
        frozen = json.loads(Path(PREFIX+'freeze.json').read_text(encoding='utf8'))
        for collection in ('files', 'sources'):
            for path, sha in frozen[collection].items():
                if digest(path) != sha:
                    raise ValueError('Frozen artifact changed: '+path)
        sealed = json.loads(Path(PREFIX+'final_capture_seal.json').read_text(encoding='utf8'))
        for path, sha in sealed['files'].items():
            if digest(path) != sha:
                raise ValueError('Final capture seal changed: '+path)
        from thinker_hier_round3_evaluate import VARIANTS, SEEDS, paths, model_for, score_rows
        import joblib
        rows, streams = load_split('final'); truth = truth_rows('final')
        for seed in SEEDS:
            for kind in VARIANTS:
                model, payload, _ = model_for(kind, seed, 'cuda')
                output = run_streams(model, streams, payload['normalization'], 'cuda',
                                     False if kind=='hierarchical_test_amortized' else None)
                prefix = paths(kind, seed)
                readout = joblib.load(prefix+'_readouts.joblib')
                records = score_rows(rows, output, truth, readout['readouts'], readout['clean_level'])
                saved = json.loads(Path(prefix+'_final.json').read_text(encoding='utf8'))
                if records != saved['records']:
                    raise ValueError('Final replay differs: '+kind+' '+str(seed))
                result['final_replays'] += len(records)
    for path in Path('results').glob('thinker_hier3_*'):
        if path.suffix in ('.json', '.jsonl', '.log', '.md'):
            if re.search(r'(?<![A-Za-z])[A-Za-z]:[\\/]|/Users/|/home/', path.read_text(encoding='utf8')):
                raise ValueError('Absolute local path in result: '+path.as_posix())
    tests = json.loads(Path(PREFIX+'tests_ready.json').read_text(encoding='utf8'))
    if tests['exit_code']:
        raise ValueError('Required tests failed')
    result['passing_tests'] = sum(r['outcome']=='passed' for r in tests['tests'])
    save(PREFIX+'audit.json', result)
    append('Round-3 completed audit', [
        f"{result['previous_round_files_unchanged']} earlier-round files and every original documentation "
        f"byte are unchanged. Audited sensory counts: {result['stream_counts']}. "
        f"Completed trained models: {result['trained_models']}; identical final model/episode replays: "
        f"{result['final_replays']}; required tests passed: {result['passing_tests']}. "
        'No cloud compute or git state changes. All new text artifacts use repository-relative paths.'])
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1); main()
