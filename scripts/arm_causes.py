"""Own-evidence arm cause learning, paired frozen scoring and checkpointed runs.

Only labelled development data enter fitting. Simulator logs are attached after
the inference trace has been captured. Large result payloads are gzip JSON.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from itertools import product

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import numpy as np
import torch
from pai.agents import slice_agent
from pai.causes.inference import Calibration, StepEvidence, SurpriseMonitor, calibrate
from pai.causes.learned import CAUSES, KNOWN, LearnedArmLikelihood, LearnedEpisodeMonitor, extract_features
from pai.config import load_config
from pai.envs import TabletopEnv
from pai.world.model import load_world_model
from slice_eval import disturbance_for


def load(path):
    p = Path(path)
    raw = gzip.decompress(p.read_bytes()) if p.suffix == '.gz' else p.read_bytes()
    return json.loads(raw)


def save(path, data, overwrite=False):
    p = Path(path)
    if p.exists() and not overwrite:
        raise FileExistsError(p)
    blob = json.dumps(data, separators=(',', ':'), allow_nan=False).encode()
    if len(blob) > 5_000_000 and p.suffix != '.gz':
        raise ValueError('Results larger than 5 MB require a .json.gz output')
    if p.suffix == '.gz':
        blob = gzip.compress(blob, mtime=0)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_bytes(blob)
    os.replace(tmp, p)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    paths = ['pai/causes/inference.py', 'pai/causes/learned.py', 'pai/agents/slice_agent.py',
             'pai/world/model.py', 'pai/perception/camera_frame.py', 'pai/perception/slot_tokens.py',
             'pai/envs/tabletop.py', 'pai/envs/events.py', 'configs/tabletop.yaml',
             'scripts/slice_eval.py', 'scripts/arm_causes.py']
    return {p: sha(ROOT / p) for p in paths}


def provenance(args):
    return {'command': [sys.executable, *sys.argv], 'python': sys.version,
            'numpy': np.__version__, 'torch': torch.__version__,
            'pythonhashseed': os.environ.get('PYTHONHASHSEED'), 'sources': sources(),
            'source_text': {p: (ROOT/p).read_bytes().decode('utf-8') for p in sources()},
            'world_model_sha256': sha(args.world_model),
            'slots_sha256': sha(args.slots) if args.regime != 'exact' else None}


def inventory():
    """Inventory explicit seed fields; older slice files omit seeds (documented)."""
    result = {}
    def walk(x, name=''):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, k)
        elif isinstance(x, list):
            for v in x:
                walk(v, name)
        elif isinstance(x, int) and not isinstance(x, bool) and 'seed' in name.lower():
            seen.add(x)
    paths = []
    for p in (ROOT/'results').iterdir():
        if p.name.startswith(('slice_v', 'pixel_v', 'language_causes_', 'arm_causes_')):
            paths.extend(p.rglob('*.json') if p.is_dir() else [p] if p.suffix in ('.json', '.gz') else [])
    for p in paths:
        seen = set()
        try:
            walk(load(p))
            result[str(p.relative_to(ROOT))] = sorted(seen)
        except (OSError, ValueError):
            result[str(p.relative_to(ROOT))] = 'unreadable'
    return result


def encode_calibration(c):
    return {'floor': c.floor.tolist(), 'threshold': c.threshold,
            'step_floors': {k: v.tolist() for k, v in c.step_floors.items()},
            'step_thresholds': c.step_thresholds}


def decode_calibration(d):
    return Calibration(np.asarray(d['floor']), d['threshold'],
                       {k: np.asarray(v) for k, v in d['step_floors'].items()}, d['step_thresholds'])


def history(record):
    # Initial collection schema omitted dtypes. Its residual/actions are exactly
    # float32-representable (verified across every development trace); sigma is
    # float64. Preserve this because the analytic estimator updates residuals
    # in place, and float32 versus float64 arithmetic can affect close fits.
    return [StepEvidence(e['t'], np.asarray(e['residual'],dtype=e.get('residual_dtype','<f4')),
                         np.asarray(e['sigma'],dtype=e.get('sigma_dtype','<f8')),
                         np.asarray(e['action'],dtype=e.get('action_dtype','<f4')),
                         e['holding'], e['step']) for e in record['own']['history']]


def encode_history(hist):
    return [{'t': int(e.t), 'residual': e.residual.tolist(), 'sigma': e.sigma.tolist(),
             'action': e.action.tolist(), 'holding': bool(e.holding), 'step': e.step,
             'residual_dtype':e.residual.dtype.str,'sigma_dtype':e.sigma.dtype.str,
             'action_dtype':e.action.dtype.str} for e in hist]


def report_dict(report):
    return asdict(report) if report is not None else None


@contextmanager
def capture_monitor(checkpoint=None):
    original = slice_agent.SurpriseMonitor
    captured = []
    def factory(*args, **kwargs):
        if checkpoint is None:
            mon = SurpriseMonitor(*args, **kwargs)
        else:
            mon = LearnedEpisodeMonitor(*args,
                likelihood=LearnedArmLikelihood.from_dict(checkpoint['terminal']['model']),
                online_likelihood=LearnedArmLikelihood.from_dict(checkpoint['online']['model']), **kwargs)
        captured.append(mon)
        return mon
    slice_agent.SurpriseMonitor = factory
    try:
        yield captured
    finally:
        slice_agent.SurpriseMonitor = original


def setup(args):
    torch.set_num_threads(1)
    cfg = load_config(args.config)
    cfg.slice.world_model = args.world_model
    cfg.slice.seed_planner = True
    cfg.env.render_images = args.regime != 'exact'
    perception = None
    if args.regime != 'exact':
        cfg.env.image_size = int(torch.load(args.slots, map_location='cpu', weights_only=False).get('image_size', 224))
    wm = load_world_model(args.world_model, args.device)
    env = TabletopEnv(cfg.env, disturbances=[])
    if args.regime != 'exact':
        from pai.perception.slot_tokens import SlotPerception
        cache = Path(torch.hub.get_dir())
        repo = cache/'facebookresearch_dinov2_main'
        weight = cache/'checkpoints/dinov2_vits14_pretrain.pth'
        if not repo.is_dir() or not weight.is_file():
            raise RuntimeError('Cached local DINOv2 code and weights required; no downloads are performed')
        old_load = torch.hub.load
        def local_load(repository, variant, *a, **kw):
            if repository == 'facebookresearch/dinov2':
                return old_load(str(repo), variant, *a, source='local', **kw)
            return old_load(repository, variant, *a, **kw)
        torch.hub.load = local_load
        try:
            perception = SlotPerception(env, args.slots, device=args.device,
                refine=args.regime == 'pixel_refined', bump_follows=True)
        finally:
            torch.hub.load = old_load
    return cfg, env, wm, perception


def actual_cause(events):
    kinds = {'payload': 'heavier_object', 'friction': 'slippery_object'}
    causes = {kinds.get(e['disturbance'], e['disturbance']) for e in events}
    if len(causes) > 1:
        raise ValueError('This benchmark assumes at most one injected cause')
    return next(iter(causes)) if causes else 'none'


def calibration_run(args):
    cfg, env, wm, perceive = setup(args)
    raw = []
    start = provenance(args)
    for i in range(args.count):
        slice_agent.run_episode(env, wm, cfg, i, args.seed+i, device=args.device, raw=raw, perceive=perceive)
        print(f'calibration {args.regime} {i+1}/{args.count}', flush=True)
    c = calibrate(np.stack([r[0] for r in raw]), np.stack([r[1] for r in raw]), [r[2] for r in raw])
    save(args.output, {'kind': 'clean_calibration', 'regime': args.regime, 'seeds': list(range(args.seed,args.seed+args.count)),
                       'calibration': encode_calibration(c), 'config': cfg.to_dict(), 'samples': len(raw),
                       'provenance': start, 'source_changes_during_run': sources() != start['sources']})
    env.close()


def collect(args):
    cfg, env, wm, perceive = setup(args)
    calibration_data = load(args.calibration)
    if calibration_data['regime'] != args.regime:
        raise ValueError('Calibration regime mismatch')
    c = decode_calibration(calibration_data['calibration'])
    frozen = load(args.model) if args.model else None
    if frozen:
        if frozen['regime'] != args.regime or frozen['frozen_sources'] != sources():
            raise ValueError('Frozen sources or regime mismatch')
        if frozen['calibration_sha256'] != sha(args.calibration):
            raise ValueError('Calibration changed')
        used = set(frozen['development_seeds']) | set(calibration_data['seeds'])
        proposed = set(range(args.seed,args.seed+args.count))
        recorded = {s for v in frozen['seed_inventory'].values() if isinstance(v,list) for s in v}
        if proposed & (used | recorded):
            raise ValueError('Evaluation seed previously used')
    if args.controller == 'learned' and frozen is None:
        raise ValueError('Learned controller requires frozen model')
    start = provenance(args)
    identity = {'seed_start':args.seed,'count':args.count,'disturbance_seed':args.disturbance_seed,
                'regime':args.regime,'controller':args.controller,'calibration_sha256':sha(args.calibration),
                'model_sha256':sha(args.model) if args.model else None,
                'world_model_sha256':start['world_model_sha256'],'slots_sha256':start['slots_sha256'],
                'config':cfg.to_dict()}
    if Path(args.output).exists():
        if not args.resume:
            raise FileExistsError(args.output)
        result = load(args.output)
        if result['provenance']['sources'] != start['sources'] or result.get('identity') != identity:
            raise ValueError('Resume requires identical seeds, controller and source')
    else:
        result = {'kind': 'arm_sensor_episodes', 'regime': args.regime, 'controller': args.controller,
                  'seed_start': args.seed, 'count': args.count, 'config': cfg.to_dict(),
                  'provenance': start, 'calibration_sha256': sha(args.calibration),
                  'model_sha256': sha(args.model) if args.model else None,
                  'identity':identity,'complete': False, 'episodes': []}
    done = {(r['seed'],r['scheduled_condition']) for r in result['episodes']}
    for seed in range(args.seed,args.seed+args.count):
        for index, condition in enumerate(KNOWN):
            if (seed,condition) in done:
                continue
            rng = np.random.default_rng(np.random.SeedSequence([args.disturbance_seed,seed,index]))
            disturbances = disturbance_for(condition,rng)
            t0 = time.time()
            monitors=[]
            try:
                with capture_monitor(frozen if args.controller == 'learned' else None) as monitors:
                    rec = slice_agent.run_episode(env,wm,cfg,len(result['episodes']),seed,disturbances,
                        device=args.device,calibration=c,perceive=perceive)
            except Exception as exc:
                result.setdefault('interruptions',[]).append({'seed':seed,'condition':condition,
                    'error':repr(exc),'traceback':traceback.format_exc(),
                    'partial_own_history':encode_history(monitors[0].raw_history) if monitors else []})
                save(args.output,result,overwrite=Path(args.output).exists())
                env.close()
                raise
            mon, = monitors
            old = SurpriseMonitor.episode_verdict(mon)
            row = {'seed': seed, 'scheduled_condition': condition,
                   'own': {'history': encode_history(mon.raw_history),
                           'trigger_steps': [int(r.t_end) for r in mon.reports]},
                   'baseline': report_dict(old),
                   'controller_terminal': {'cause': rec.inferred_cause, 'posterior': rec.cause_posterior,
                                           'params': rec.cause_params},
                   'adaptations': rec.adaptations,
                   'withheld_adaptations': getattr(mon,'withheld_adaptations',[]),
                   'scoring': {'cause': actual_cause(rec.true_disturbances), 'success': rec.success,
                               'events': rec.events, 'disturbances': rec.true_disturbances},
                   'wall_seconds': time.time()-t0}
            result['episodes'].append(row)
            save(args.output,result,overwrite=Path(args.output).exists())
            print(f"{args.regime} {args.controller} {seed} {condition}: old={old.best if old else 'none'} task={rec.success} {row['wall_seconds']:.1f}s",flush=True)
    result['complete'] = True
    result['source_changes_during_run'] = sources() != start['sources']
    save(args.output,result,overwrite=True)
    env.close()


def terminal_rows(episodes,dt):
    return [{'features':extract_features(history(r),dt,bool(r['own']['trigger_steps']))[0],
             'label':r['scoring']['cause'],'seed':r['seed'],'weight':1.} for r in episodes]


def prefix_rows(episodes,dt,repeat):
    out = []
    for r in episodes:
        hist = history(r)
        stops = sorted(set(range(10,len(hist)+1,10)) | {len(hist)} |
                       {i+1 for i,e in enumerate(hist) if e.t in r['own']['trigger_steps']})
        for n in stops:
            last = hist[n-1].t
            events = [e for e in r['scoring']['disturbances'] if e['t'] <= (last+1)*repeat]
            triggered = any(t <= last for t in r['own']['trigger_steps'])
            out.append({'features':extract_features(hist[:n],dt,triggered)[0],
                        'label':actual_cause(events),'seed':r['seed'],'weight':1./len(stops)})
    return out


def metrics(labels,predictions):
    n = len(labels);k = sum(a==b for a,b in zip(labels,predictions))
    return {'n':n,'correct':k,'accuracy':k/n,
            'by_cause':{c:{'n':sum(a==c for a in labels),'correct':sum(a==c and b==c for a,b in zip(labels,predictions))}
                        for c in KNOWN},
            'confusion':[[sum(a==c and b==q for a,b in zip(labels,predictions)) for q in CAUSES] for c in KNOWN]}


def fit_rows(rows):
    candidates=[]
    seeds=sorted({r['seed'] for r in rows})
    for feature_set in ('raw','fits','fused'):
        for shrinkage,family,prior_kind in product((.5,.9,1.),('gaussian','student_t4'),('uniform','legacy')):
            labels=[];predictions=[];folds=[]
            for seed in seeds:
                train=[r for r in rows if r['seed']!=seed];valid=[r for r in rows if r['seed']==seed]
                m=LearnedArmLikelihood.fit([r['features'] for r in train],[r['label'] for r in train],
                    feature_set,shrinkage,weights=[r['weight'] for r in train],
                    likelihood_family=family,prior_kind=prior_kind)
                y=[r['label'] for r in valid]
                p=[]
                for r in valid:
                    post=m.posterior(r['features']);p.append(max(post,key=post.get))
                labels+=y;predictions+=p;folds.append({'seed':seed,**metrics(y,p)})
            score=metrics(labels,predictions)
            score['balanced_accuracy']=float(np.mean([v['correct']/v['n'] for v in score['by_cause'].values() if v['n']]))
            score['false_pushes_in_none_heavy_slip']=sum(score['confusion'][i][1] for i in (0,2,3))
            candidates.append({'feature_set':feature_set,'shrinkage':shrinkage,'family':family,'prior_kind':prior_kind,
                               'folds':folds,'score':score})
    best=max(candidates,key=lambda r:(r['score']['balanced_accuracy'],-r['score']['false_pushes_in_none_heavy_slip']))
    m=LearnedArmLikelihood.fit([r['features'] for r in rows],[r['label'] for r in rows],
        best['feature_set'],best['shrinkage'],weights=[r['weight'] for r in rows],
        likelihood_family=best['family'],prior_kind=best['prior_kind'])
    return {'selected':{k:best[k] for k in ('feature_set','shrinkage','family','prior_kind','score')},
            'candidates':candidates,'model':m.to_dict()}


def fit(args):
    datasets=[load(p) for p in args.inputs]
    if not all(d['complete'] and not d['source_changes_during_run'] and d['controller']=='baseline' for d in datasets):
        raise ValueError('Fitting requires complete unchanged-source baseline development records')
    if {d['regime'] for d in datasets}!={args.regime}:
        raise ValueError('Fit each perception regime independently')
    calibration_data=load(args.calibration)
    if any(d['calibration_sha256']!=sha(args.calibration) for d in datasets):
        raise ValueError('Development calibration mismatch')
    episodes=[r for d in datasets for r in d['episodes']]
    cfg=datasets[0]['config'];repeat=int(cfg['slice']['action_repeat']);dt=cfg['env']['control_dt']*repeat
    terminal=terminal_rows(episodes,dt);online=prefix_rows(episodes,dt,repeat)
    result={'kind':'frozen_learned_arm_likelihoods','regime':args.regime,
            'development_seeds':sorted({r['seed'] for r in episodes}),
            'input_sha256':{str(p):sha(p) for p in args.inputs},'calibration_sha256':sha(args.calibration),
            'config':cfg,'seed_inventory':inventory(),'frozen_sources':sources(),
            'world_model_sha256':datasets[0]['provenance']['world_model_sha256'],
            'slots_sha256':datasets[0]['provenance']['slots_sha256'],
            'terminal':fit_rows(terminal),'online':fit_rows(online),
            'training_counts':{'terminal':len(terminal),'online_prefixes':len(online)},
            'source_text':{p:(ROOT/p).read_bytes().decode('utf-8') for p in sources()}}
    save(args.output,result)
    print(json.dumps({k:result[k]['selected'] for k in ('terminal','online')},indent=2),flush=True)


def wilson(k,n):
    if not n:return [None,None]
    z=1.959963984540054;p=k/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den;delta=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0,mid-delta),min(1,mid+delta)]


def parameter_errors(row,cause,params,dt,control_dt):
    if cause!=row['scoring']['cause']:return {}
    events=row['scoring']['disturbances'];matched=[e for e in events if actual_cause([e])==cause]
    prm=params.get(cause,{}) if params else {}
    out={}
    if cause=='heavier_object':
        out['mass_estimate_available']='extra_mass_kg' in prm
        if matched and 'extra_mass_kg' in prm:out['mass_abs_error_kg']=abs(prm['extra_mass_kg']-matched[0]['params']['mass'])
    if cause=='camera_shift':
        out['offset_estimate_available']='camera_offset_mm' in prm
        if matched and 'camera_offset_mm' in prm:
            out['offset_error_mm']=float(np.linalg.norm(np.asarray(prm['camera_offset_mm'])-1000*np.asarray(matched[0]['params']['offset'])))
    if cause in ('push','camera_shift','slippery_object'):
        out['onset_estimate_available']='onset' in prm
        target=matched[0]['t']*control_dt if matched else None
        if cause=='slippery_object':
            slips=[e for e in row['scoring']['events'] if e['type']=='slip' and e.get('object')=='red']
            target=slips[0]['t']*control_dt if slips else None
        if target is not None and 'onset' in prm:out['onset_error_s']=abs(prm['onset']*dt-target)
        else:out['onset_matching_event_available']=target is not None
    return out


def score(args):
    data=load(args.input);frozen=load(args.model)
    if not data['complete'] or data['source_changes_during_run']:raise ValueError('Incomplete or changed-source evaluation')
    if data['regime']!=frozen['regime'] or sources()!=frozen['frozen_sources']:raise ValueError('Frozen model mismatch')
    if set(r['seed'] for r in data['episodes']) & set(frozen['development_seeds']):raise ValueError('Development seeds in final evaluation')
    if any(data['provenance'][k]!=frozen[k] for k in ('world_model_sha256','slots_sha256')):raise ValueError('Checkpoint changed')
    m=LearnedArmLikelihood.from_dict(frozen['terminal']['model'])
    repeat=data['config']['slice']['action_repeat'];control_dt=data['config']['env']['control_dt'];dt=repeat*control_dt
    rows=[]
    for r in data['episodes']:
        new=m.infer(history(r),dt,bool(r['own']['trigger_steps']));old=r['baseline']
        old_cause=max(old['posterior'],key=old['posterior'].get) if old else 'none'
        rows.append({'seed':r['seed'],'scheduled_condition':r['scheduled_condition'],'truth':r['scoring']['cause'],
                     'success':r['scoring']['success'],'old_cause':old_cause,'new_cause':new.best,
                     'new_posterior':new.posterior,'new_params':new.params,
                     'old_errors':parameter_errors(r,old_cause,old['params'] if old else {},dt,control_dt),
                     'new_errors':parameter_errors(r,new.best,new.params,dt,control_dt)})
    summaries={}
    for method in ('old','new'):
        s=metrics([r['truth'] for r in rows],[r[method+'_cause'] for r in rows])
        s['wilson95']=wilson(s['correct'],s['n'])
        for c,v in s['by_cause'].items():
            v['recall']=v['correct']/v['n'] if v['n'] else None;v['wilson95']=wilson(v['correct'],v['n'])
            own=[r[method+'_errors'] for r in rows if r['truth']==c and r[method+'_cause']==c]
            v['parameters']={}
            for field,tol in [('mass_abs_error_kg',.1),('offset_error_mm',10.),('onset_error_s',.3)]:
                values=[e[field] for e in own if field in e]
                if values:
                    v['parameters'][field]={'correct_cause_n':len(own),'available_n':len(values),
                        'mean':float(np.mean(values)),'median':float(np.median(values)),
                        'tolerance':tol,'within_tolerance':sum(e<=tol for e in values)}
        for field,tol in [('mass_abs_error_kg',.1),('offset_error_mm',10.),('onset_error_s',.3)]:
            values=[r[method+'_errors'][field] for r in rows if field in r[method+'_errors']]
            s[field]={'n':len(values),'mean':float(np.mean(values)) if values else None,
                      'median':float(np.median(values)) if values else None,'tolerance':tol,
                      'within_tolerance':sum(v<=tol for v in values)}
        summaries[method]=s
    rng=np.random.default_rng(870000);seeds=sorted({r['seed'] for r in rows})
    gains=[]
    for _ in range(5000):
        sampled=rng.choice(seeds,len(seeds));rs=[r for seed in sampled for r in rows if r['seed']==seed]
        gains.append(np.mean([r['new_cause']==r['truth'] for r in rs])-np.mean([r['old_cause']==r['truth'] for r in rs]))
    new_only=sum(r['new_cause']==r['truth'] and r['old_cause']!=r['truth'] for r in rows)
    old_only=sum(r['old_cause']==r['truth'] and r['new_cause']!=r['truth'] for r in rows)
    result={'kind':'paired_arm_cause_gate','regime':data['regime'],'controller':data['controller'],
            'input_sha256':sha(args.input),'model_sha256':sha(args.model),'summary':summaries,
            'paired':{'new_only_correct':new_only,'old_only_correct':old_only,
                      'accuracy_gain':summaries['new']['accuracy']-summaries['old']['accuracy'],
                      'layout_bootstrap95':np.quantile(gains,[.025,.975]).tolist(),'bootstrap_seed':870000},
            'rows':rows,'inference_sources':sources()}
    save(args.output,result)
    print(json.dumps({'regime':data['regime'],'old':summaries['old']['accuracy'],'new':summaries['new']['accuracy'],'paired':result['paired']},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('inventory','calibrate','collect','fit','score'))
    p.add_argument('--config',default='configs/tabletop.yaml')
    p.add_argument('--world-model',default='runs/slice_world_model/world_model.pt')
    p.add_argument('--slots',default='runs/slots/slots.pt')
    p.add_argument('--regime',choices=('exact','pixel_slots','pixel_refined'),default='exact')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed',type=int,default=820000)
    p.add_argument('--count',type=int,default=5)
    p.add_argument('--disturbance-seed',type=int,default=840000)
    p.add_argument('--controller',choices=('baseline','learned'),default='baseline')
    p.add_argument('--calibration',type=Path)
    p.add_argument('--model',type=Path)
    p.add_argument('--input',type=Path)
    p.add_argument('--inputs',type=Path,nargs='+')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.output.exists() and not(args.mode=='collect' and args.resume):raise FileExistsError(args.output)
    if args.mode=='inventory':save(args.output,inventory())
    else:{'calibrate':calibration_run,'collect':collect,'fit':fit,'score':score}[args.mode](args)


if __name__=='__main__':main()
