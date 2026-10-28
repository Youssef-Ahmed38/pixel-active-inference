"""Identical round-1 world-model training budget on round-3 label-free streams."""
import argparse
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from pai.thinker.predictive import PredictiveConfig, parameters, predictive_loss
from pai.thinker.hier_data import normalization, batch_windows
from pai.thinker.hier3_tokens import load_split
from thinker_hier_train import build, KINDS
from thinker_hier_collect import digest
from thinker_hier_round2 import save
from thinker_hier_round3 import register, PREFIX
from thinker_hier_round3_collect import require_gate

SEEDS = [100000, 100001, 100002]


def source_hashes():
    return {p.as_posix(): digest(p) for pattern in ('pai/thinker/predictive.py',
        'pai/thinker/hier_analysis.py', 'pai/thinker/hier_data.py', 'pai/thinker/hier3_tokens.py',
        'scripts/thinker_hier_round3*.py') for p in Path('.').glob(pattern)}


def train(kind, seed, device):
    plan = register()
    require_gate('gate_a'); require_gate('gate_b')
    if seed not in SEEDS or kind not in KINDS:
        raise ValueError('Use registered kind/optimizer seed')
    rows, streams = load_split('train')
    if len(streams) != 1000:
        raise ValueError('All registered training streams required')
    cfg = PredictiveConfig(**plan['model'])
    torch.set_num_threads(1); torch.manual_seed(seed)
    model = build(kind, cfg).to(device)
    if parameters(model) != plan['parameters'][kind]:
        raise ValueError('Parameter budget changed')
    norm = normalization(streams)
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan['lr'], weight_decay=.0001)
    rng = np.random.default_rng(seed); start = time.time(); history = []
    prefix = PREFIX+f'{kind}_s{seed}'
    output = Path(prefix+'.pt')
    if output.exists():
        raise FileExistsError(output)
    sources = source_hashes()
    for step in range(plan['steps']):
        s, a, m, y = batch_windows(streams, norm, rng, plan['batch_size'], plan['sequence_length'], device)
        model.train(); optimizer.zero_grad(set_to_none=True)
        prediction, trace = model(s, a, m)
        loss = predictive_loss(prediction, y, model)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite loss; keep failed variant')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step%100 == 0 or step+1 == plan['steps']:
            history.append(dict(step=step+1, nll=float(loss.detach()),
                mse=float((prediction-y).square().mean().detach()),
                iterations=float(trace['iterations'].float().mean()), elapsed_s=time.time()-start))
            print(kind, seed, history[-1], flush=True)
    payload = dict(kind=kind, seed=seed, model=model.state_dict(), config=asdict(cfg), normalization=norm,
                   plan_sha256=digest(PREFIX+'plan.json'), source_hashes=sources)
    torch.save(payload, output)
    save(prefix+'_training.json', dict(kind=kind, seed=seed, parameters=parameters(model), steps=plan['steps'],
        train_wall_s=time.time()-start, history=history, checkpoint=output.as_posix(), checkpoint_sha256=digest(output),
        source_hashes=sources, streams={r['path']: r['sha256'] for r in rows}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kind', choices=KINDS, required=True)
    ap.add_argument('--seed', type=int, required=True); ap.add_argument('--device', default='cuda')
    args = ap.parse_args(); train(args.kind, args.seed, args.device)
