"""Label-free stream loading and normalization; deliberately rejects scoring fields."""
import json
from pathlib import Path
import numpy as np
import torch

ALLOWED={'state','action','context','tokens','q','dq'}

def load_stream(path):
    with np.load(path,allow_pickle=False) as data:
        if set(data.files)!=ALLOWED:raise ValueError('Unexpected field in sensory stream')
        return {k:data[k].copy() for k in ('state','action','context')}

def load_split(split):
    manifests=sorted(Path('results').glob(f'thinker_hier_{split}_manifest*.json'))
    if not manifests:raise ValueError('No stream manifests')
    streams=[];rows=[]
    for path in manifests:
        manifest=json.loads(path.read_text(encoding='utf8'))
        if not manifest['complete']:raise ValueError('Collection incomplete')
        for r in manifest['episodes']:
            rows.append(r);streams.append(load_stream(r['path']))
    order=np.argsort([r['index'] for r in rows]);rows=[rows[i] for i in order];streams=[streams[i] for i in order]
    if len({r['index'] for r in rows})!=len(rows):raise ValueError('Duplicate episodes')
    return rows,streams

def normalization(streams):
    # Training senses only, no episode identifiers/condition metadata.
    s=np.concatenate([x['state'] for x in streams]);a=np.concatenate([x['action'] for x in streams])
    return dict(s_mean=s.mean(0).tolist(),s_std=np.maximum(s.std(0),1e-3).tolist(),
                a_mean=a.mean(0).tolist(),a_std=np.maximum(a.std(0),1e-3).tolist())

def normalized(stream,norm):
    return dict(state=(stream['state']-norm['s_mean'])/norm['s_std'],
                action=(stream['action']-norm['a_mean'])/norm['a_std'],context=stream['context'])

def batch_windows(streams,norm,rng,batch=16,length=24,device='cuda'):
    rows=[]
    for index in rng.integers(len(streams),size=batch):
        r=normalized(streams[index],norm);n=len(r['action'])
        start=int(rng.integers(max(1,n-length+1)));ids=np.minimum(np.arange(start,start+length),n-1)
        rows.append((r['state'][ids],r['action'][ids],r['context'][ids],r['state'][ids+1]))
    return [torch.as_tensor(np.stack([r[j] for r in rows]),dtype=torch.float32,device=device) for j in range(4)]
