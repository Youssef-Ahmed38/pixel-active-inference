"""Normal privileged-token episodes, adaptation off, gated before collection."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from pai.agents.slice_agent import run_episode
from pai.causes.inference import Calibration, calibrate
from pai.config import load_config
from pai.envs import TabletopEnv
from pai.thinker.hier3_tokens import RecordingDefaultPerception, sensory_vector, truth_rows
from pai.thinker.hier_analysis import rate
from pai.world.model import load_world_model
from slice_eval import CONDITIONS, disturbance_for
from thinker_hier_collect import digest
from thinker_hier_round2 import save
from thinker_hier_round3 import register, PREFIX


def require_gate(name):
    status = Path(PREFIX+name+'_status.json')
    path = status if status.exists() else Path(PREFIX+name+'.json')
    if not json.loads(path.read_text(encoding='utf8'))['passed']:
        raise ValueError('Failed '+name+'; collection/training prohibited')


def calibration_value():
    v = json.loads(Path(PREFIX+'reference_calibration.json').read_text(encoding='utf8'))
    return Calibration(floor=np.array(v['floor']), threshold=v['threshold'],
        step_floors={k: np.array(x) for k, x in v['step_floors'].items()}, step_thresholds=v['step_thresholds'])


def collect(split, worker, workers, device):
    plan = register()
    require_gate('gate_a')
    if split in ('train', 'final'):
        require_gate('gate_b')
    if split == 'final' and not Path(PREFIX+'freeze.json').exists():
        raise ValueError('Final collection requires frozen models/readouts')
    cfg = load_config('configs/tabletop.yaml', ['env.render_images=false'])
    if cfg.slice.to_dict() != plan['reproduction_config']['slice']:
        raise ValueError('Normal motor configuration changed')
    if digest(cfg.slice.world_model) != plan['world_model_sha256']:
        raise ValueError('Registered arm world model changed')
    for path, sha in plan['registered_sources'].items():
        if digest(path) != sha:
            raise ValueError('Registered arm source changed: '+path)
    wm = load_world_model(cfg.slice.world_model, device)
    env = TabletopEnv(cfg.env, disturbances=[])
    perceive = RecordingDefaultPerception(env)
    c = Calibration() if split == 'calibration' else calibration_value()
    prefix = PREFIX+split
    mp, tp = Path(prefix+f'_manifest{worker}.json'), Path(prefix+f'_truth{worker}.json')
    sources = {p: digest(p) for p in ('scripts/thinker_hier_round3_collect.py',
        'pai/thinker/hier3_tokens.py', 'pai/agents/slice_agent.py', 'pai/perception/camera_frame.py',
        'pai/causes/inference.py', 'configs/tabletop.yaml')}
    meta = dict(split=split, worker=worker, workers=workers, source_hashes=sources,
        world_model_sha256=digest(cfg.slice.world_model), episodes=[], complete=False)
    truth = dict(split=split, episodes=[])
    if mp.exists():
        old = json.loads(mp.read_text(encoding='utf8'))
        for key in ('workers', 'source_hashes', 'world_model_sha256'):
            if old[key] != meta[key]:
                raise ValueError('Cannot resume changed collection: '+key)
        meta = old; truth = json.loads(tp.read_text(encoding='utf8'))
    seeds = plan['data_seeds'][split]
    try:
        for ci, condition in enumerate(['none'] if split == 'calibration' else CONDITIONS):
            for i, seed in enumerate(seeds):
                index = ci*len(seeds)+i
                if index % workers != worker or any(r['index'] == index for r in meta['episodes']):
                    continue
                start = time.time()
                raw, evidence, sensory = [], [], []
                disturbance = disturbance_for(condition, np.random.default_rng(
                    plan['disturbance_rng_bases'].get(split, 1030000)+ci*1000+i))
                record = run_episode(env, wm, cfg, index, seed, disturbance, device=device,
                    perceive=perceive, adapt=False, calibration=c, raw=raw, evidence=evidence, sensory=sensory)
                n = len(raw)
                if len(perceive.tokens) != n+1 or len(evidence) != n or len(sensory) != len(record.actions):
                    raise ValueError('Incomplete plan/settle capture')
                if any(not np.array_equal(s['after'], t) for s, t in zip(sensory, perceive.tokens[1:])):
                    raise ValueError('Recorder changed default tokens')
                action = np.concatenate([record.actions,
                    np.tile([0., 0., 0., 1.], (n-len(record.actions), 1))]).astype(np.float32)
                holding = np.array([r[1] for r in evidence]); phase = np.array([r[2] for r in evidence])
                path = Path(prefix+f'_{index:04d}.npz')
                np.savez_compressed(path, state=np.stack([sensory_vector(t) for t in perceive.tokens]),
                    action=action, context=np.c_[holding, np.eye(9)[np.minimum(phase, 8)]],
                    tokens=np.stack(perceive.tokens), q=np.stack([r[0] for r in perceive.proprio[1:]]),
                    dq=np.stack([r[1] for r in perceive.proprio[1:]]))
                ref = Path(prefix+f'_reference_{index:04d}.npz')
                np.savez_compressed(ref, residual=np.stack([r[0] for r in raw]),
                    sigma=np.stack([r[1] for r in raw]), step=np.array([r[2] for r in raw]),
                    action=action, holding=holding, phase=phase)
                row = dict(index=index, seed=seed, assigned_condition=condition,
                    disturbances=record.true_disturbances, events=record.events, success=record.success,
                    inferred=record.inferred_cause, posterior=record.cause_posterior,
                    reference=ref.as_posix(), holding_steps=int(holding.sum()), steps=n,
                    settle_steps=sum(r[2]=='settle' for r in raw), collection_s=time.time()-start)
                truth['episodes'] = [r for r in truth['episodes'] if r['index'] != index]+[row]
                meta['episodes'].append(dict(index=index, path=path.as_posix(), sha256=digest(path), steps=n))
                save(tp, truth); save(mp, meta)
                print(split, index, condition, seed, n, record.success, record.inferred_cause,
                    'seconds', round(time.time()-start, 1), flush=True)
        meta['complete'] = True; save(mp, meta)
    finally:
        env.close()


def calibrate_reference():
    data = truth_rows('calibration')
    if len(data) != 8:
        raise ValueError('All eight independent calibration episodes required')
    res, sig, names = [], [], []
    for row in data.values():
        with np.load(row['reference'], allow_pickle=False) as d:
            res.extend(d['residual']); sig.extend(d['sigma']); names.extend(d['step'])
    c = calibrate(np.stack(res), np.stack(sig), names)
    save(PREFIX+'reference_calibration.json', dict(floor=c.floor.tolist(), threshold=c.threshold,
        step_floors={k: v.tolist() for k, v in c.step_floors.items()}, step_thresholds=c.step_thresholds,
        samples=len(res), seeds=register()['data_seeds']['calibration']))


def gate_b():
    records = list(truth_rows('development').values())
    if len(records) != 200:
        raise ValueError('All 200 development episodes required')
    correct = sum(r['inferred']==r['assigned_condition'] for r in records)
    result = dict(passed=correct>=160, accuracy=rate(correct, 200),
        placement=rate(sum(r['success'] for r in records), 200), per_condition={}, records=records)
    for condition in CONDITIONS:
        rs = [r for r in records if r['assigned_condition']==condition]
        result['per_condition'][condition] = dict(
            accuracy=rate(sum(r['inferred']==condition for r in rs), len(rs)),
            placement=rate(sum(r['success'] for r in rs), len(rs)),
            grasp_episodes=sum(any(e['type']=='grasp' for e in r['events']) for r in rs),
            holding_episodes=sum(r['holding_steps']>0 for r in rs),
            realized_disturbance_episodes=sum(bool(r['disturbances']) for r in rs),
            full_post20_episodes=sum(any(r['steps']-max(0,(int(e['t'])-1)//5)>=20
                                       for e in r['disturbances']) for r in rs))
    save(PREFIX+'gate_b.json', result)
    save(PREFIX+'gate_b_status.json', dict(passed=result['passed'], gate_sha256=digest(PREFIX+'gate_b.json')))
    print(json.dumps({k: v for k, v in result.items() if k!='records'}, indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['collect', 'calibrate', 'gate-b'])
    ap.add_argument('--split', choices=['calibration', 'development', 'train', 'final'])
    ap.add_argument('--worker', type=int, default=0); ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args(); torch.set_num_threads(1)
    if args.stage == 'collect': collect(args.split, args.worker, args.workers, args.device)
    elif args.stage == 'calibrate': calibrate_reference()
    else: gate_b()
