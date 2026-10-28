"""Collect, fit, and evaluate sensor-only fall likelihoods with frozen locomotion."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch

from pai.config import load_config
from pai.locomotion.env import EnvConfig,LocomotionEnv
from pai.locomotion.policy import LocomotionController
from pai.locomotion.world_model import EnsembleBodyModel
from pai.locomotion.falls import Calibration,posterior as old_posterior
from pai.locomotion.fall_causes import (CAUSES,CHANNELS,instrumented_body,load_cells,FallEvidenceMonitor,
    extract_features,pack_trace,unpack_trace,LearnedFallLikelihood)
from scripts.locomotion_eval import gait_error_commands,ice_models


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,data):
    path=Path(path)
    tmp=path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)


def wilson(k,n):
    if not n:
        return dict(k=int(k),n=int(n),rate=None,wilson95=None)
    z=1.959963984540054
    p=k/n
    den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return dict(k=int(k),n=int(n),rate=float(p),wilson95=[max(0.,float(mid-half)),min(1.,float(mid+half))])


def metrics(rows,predictions):
    confusion={c:{p:0 for p in CAUSES} for c in CAUSES[:3]}
    for row,p in zip(rows,predictions):
        confusion[row['label']][p]+=1
    recalls={c:wilson(confusion[c][c],sum(confusion[c].values())) for c in CAUSES[:3]}
    return dict(accuracy=wilson(sum(r['label']==p for r,p in zip(rows,predictions)),len(rows)),
        balanced_accuracy=float(np.mean([v['rate'] for v in recalls.values() if v['rate'] is not None])),
        per_class_recall=recalls,confusion_true_by_pred=confusion,
        unknown=wilson(sum(p=='unknown' for p in predictions),len(rows)))


def feature_rows(rows,kind):
    if kind=='legacy_learned':
        return [dict(U=r['legacy_features'][0],S=r['legacy_features'][1]) for r in rows]
    return [r['features' if kind=='full' else 'tail_features'] for r in rows]


def provenance(args):
    files=['pai/locomotion/fall_causes.py','pai/locomotion/falls.py','pai/locomotion/env.py',
           'pai/locomotion/body.py','pai/locomotion/world_model.py','pai/locomotion/policy.py',
           'scripts/locomotion_fall_causes.py','scripts/locomotion_eval.py',args.config]
    return dict(command=[sys.executable,*sys.argv],commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                code_hashes={p:digest(p) for p in files},policy_sha256=digest(args.policy),
                body_model_checkpoint_sha256=digest(args.ckpt),python=sys.version,torch=torch.__version__,
                numpy=np.__version__)


def collect(args):
    import mujoco
    cfg=load_config(args.config,[])
    ecfg=EnvConfig.from_dict(cfg.env.to_dict())
    ecfg.nthread=args.threads
    ecfg.episode_s=60.
    torch.set_num_threads(1)
    ctrl=LocomotionController(Path(args.policy),args.device)
    ck=torch.load(args.ckpt,map_location='cpu',weights_only=False)
    wm=EnsembleBodyModel(ck['train_cfg']['wm_members'])
    wm.load_state_dict(ck['model'])
    wm.to(args.device).eval()
    body,addresses=instrumented_body()
    conditions=args.conditions.split(',')
    n=args.per*len(conditions)
    env=LocomotionEnv(n,ecfg,seed=args.seed,stage=len(ecfg.stages)-1,body=body)
    env.pushes_enabled=False
    env.set_command(np.arange(n),0.,0.,0.)
    env.reset(np.arange(n))
    obs=env.observe()
    start_yaw=env.yaw().copy()
    env.set_command(np.arange(n),.6,0.,0.,heading=start_yaw)
    monitor=FallEvidenceMonitor(wm,n,args.device,args.chunk,ecfg.contact_height)
    alive=np.ones(n,bool)
    labels=np.repeat(conditions,args.per)
    rng=np.random.default_rng(args.seed)
    records=[]
    pre_falls=[]
    metadata=provenance(args)
    report=dict(seed=args.seed,conditions=conditions,per_condition=args.per,config=asdict(ecfg),
                provenance=metadata,channels=CHANNELS,injected_at=args.inject_at,after_s=args.after,
                push_command=args.push_command,slip_command=args.slip_command,
                neural_chunk=args.chunk,records=records,pre_injection_falls=pre_falls,complete=False)
    report['solver_warnings']=[]
    warning_context={'step':0}
    def record_warning(message):
        report['solver_warnings'].append(dict(step=warning_context['step'],message=str(message)))
        print('solver warning:',message,flush=True)
    mujoco.set_mju_user_warning(record_warning)
    before=None
    truth_events={}
    start=time.perf_counter()
    inject_step=int(round(args.inject_at/ecfg.ctrl_dt))
    steps=int(round((args.inject_at+args.after)/ecfg.ctrl_dt))
    for t in range(steps):
        warning_context['step']=t+1
        if t==inject_step:
            before=alive.copy()
            for cause in conditions:
                ids=np.flatnonzero(labels==cause)
                if cause=='push':
                    dv=rng.uniform(1.5,2.5,len(ids))
                    angles=rng.uniform(-np.pi,np.pi,len(ids))
                    for j,i in enumerate(ids):
                        force=env.push_force_for_dv(dv[j])
                        env.apply_push([i],[force*np.cos(angles[j]),force*np.sin(angles[j]),0.])
                        truth_events[int(i)]=dict(type='push',time=args.inject_at,target_dv=float(dv[j]),duration=ecfg.push_s)
                    if args.push_command is not None:
                        # speed control: change the goal at the push exactly as slip does
                        env.set_command(ids,args.push_command,0.,0.,heading=start_yaw[ids])
                elif cause=='slip':
                    ice_models(env,ids)
                    env.set_command(ids,args.slip_command,0.,0.,heading=start_yaw[ids])
                    for i in ids:
                        truth_events[int(i)]=dict(type='slip',time=args.inject_at,friction_multiplier=.05)
                elif cause=='gait':
                    gait_error_commands(env,ids,rng)
                    for i in ids:
                        truth_events[int(i)]=dict(type='gait',time=args.inject_at)
        action=np.concatenate([ctrl.act(obs[i:i+args.chunk]) for i in range(0,n,args.chunk)])
        nobs,_,done,info=env.step(action)
        nxt=nobs.copy()
        sensors=env.sensors.copy()
        ids=info.get('done_ids',np.array([],int))
        if len(ids):
            nxt[ids]=info['final_obs']
            sensors[ids]=info['final_sensors']
        heights,velocity=env.soles(sensors)
        forces,torques=load_cells(sensors,addresses)
        monitor.update(obs,action,nxt,heights,velocity,forces,torques,active=alive)
        new=ids[info['fell'][ids] & alive[ids]] if len(ids) else []
        for i in new:
            fall_time=(t+1)*ecfg.ctrl_dt
            if t<inject_step:
                pre_falls.append(dict(body=int(i),label=str(labels[i]),fall_time=fall_time))
            else:
                # Only here attach the injection answer key, AFTER sensory extraction.
                trace=np.asarray(monitor.history[i],dtype=np.float32)
                features,onsets=extract_features(trace)
                tail,_=extract_features(trace[-100:])
                old=monitor.legacy.features([i])[0].tolist()
                records.append(dict(body=int(i),label=str(labels[i]),fall_time=fall_time,
                    features=features,tail_features=tail,onsets=onsets,legacy_features=old,
                    evidence=pack_trace(trace),truth_event=truth_events[int(i)]))
            alive[i]=False
        obs=nobs
        if (t+1)%50==0:
            report['steps_completed']=t+1
            save(args.output,report)
            print('seed',args.seed,'t',round((t+1)*ecfg.ctrl_dt,2),'scored falls',len(records),flush=True)
    report['injected_survivors']={c:int(((labels==c)&before).sum()) for c in conditions}
    report['post_injection_nonfalls']={c:int(((labels==c)&before&alive).sum()) for c in conditions}
    report['seconds_wall']=time.perf_counter()-start
    report['complete']=True
    save(args.output,report)
    mujoco.set_mju_user_warning(None)


def fit(args):
    datasets=[json.loads(Path(p).read_text(encoding='utf-8')) for p in args.inputs]
    if not all(d['complete'] for d in datasets):
        raise ValueError('Incomplete development collection')
    candidates=[]
    for kind in ('full','tail','legacy_learned'):
        for shrinkage in (.1,.5,.9):
            results=[]
            for hold in range(len(datasets)):
                train=[r for j,d in enumerate(datasets) if j!=hold for r in d['records']]
                valid=datasets[hold]['records']
                model=LearnedFallLikelihood.fit(feature_rows(train,kind),[r['label'] for r in train],shrinkage)
                p=model.posterior(feature_rows(valid,kind))
                result=metrics(valid,[CAUSES[j] for j in p.argmax(1)])
                results.append(dict(seed=datasets[hold]['seed'],**result))
            candidates.append(dict(feature_kind=kind,shrinkage=shrinkage,folds=results,
                balanced_accuracy=float(np.mean([r['balanced_accuracy'] for r in results]))))
    best=max(candidates,key=lambda c:c['balanced_accuracy'])
    rows=[r for d in datasets for r in d['records']]
    model=LearnedFallLikelihood.fit(feature_rows(rows,best['feature_kind']),[r['label'] for r in rows],best['shrinkage'])
    save(args.output,dict(development_seeds=[d['seed'] for d in datasets],
        sources={str(p):digest(p) for p in args.inputs},candidate_scores=candidates,selected=best,
        learned_model=model.to_dict(),class_counts={c:sum(r['label']==c for r in rows) for c in CAUSES[:3]},
        provenance=provenance(args),unknown_note='Broad Student-t form and prior given; pooled center/scale learned. No real unknown labels.'))
    print(json.dumps({k:best[k] for k in ('feature_kind','shrinkage','balanced_accuracy')},indent=2))


def score(args):
    data=json.loads(Path(args.input).read_text(encoding='utf-8'))
    checkpoint=json.loads(Path(args.model).read_text(encoding='utf-8'))
    calibration=json.loads(Path(args.calibration).read_text(encoding='utf-8'))
    if not data['complete'] or not calibration['complete']:
        raise ValueError('Incomplete final/calibration data')
    if data['seed'] in checkpoint['development_seeds']:
        raise ValueError('Final seed used in development')
    if calibration['seed'] in checkpoint['development_seeds'] or calibration['conditions']!=['gait']:
        raise ValueError('Calibration must be separate undisturbed gait falls')
    for key in ('policy_sha256','body_model_checkpoint_sha256'):
        if any(d['provenance'][key]!=checkpoint['provenance'][key] for d in (data,calibration)):
            raise ValueError('Policy or body model changed')
    model=LearnedFallLikelihood(**checkpoint['learned_model'])
    rows=data['records']
    probabilities=model.posterior(feature_rows(rows,checkpoint['selected']['feature_kind']))
    cal=Calibration.fit(np.array([r['legacy_features'] for r in calibration['records']]))
    old=old_posterior(np.array([r['legacy_features'] for r in rows]),cal)
    outrows=[]
    for row,p,q in zip(rows,probabilities,old):
        evidence_features,_=extract_features(unpack_trace(row['evidence']))
        if evidence_features!=row['features']:
            raise ValueError('Feature record cannot be replayed')
        outrows.append(dict(body=row['body'],truth=row['label'],fall_time=row['fall_time'],
            inferred=CAUSES[int(p.argmax())],posterior=dict(zip(CAUSES,map(float,p))),
            old_inferred=CAUSES[int(q.argmax())],old_posterior=dict(zip(CAUSES[:3],map(float,q))),
            model_features=feature_rows([row],checkpoint['selected']['feature_kind'])[0],
            whole_history_features=row['features'],whole_history_onsets=row['onsets'],truth_event=row['truth_event']))
    result=dict(seed=data['seed'],calibration_seed=calibration['seed'],
        final_data=str(args.input),final_data_sha256=digest(args.input),model_sha256=digest(args.model),
        calibration_sha256=digest(args.calibration),old_calibration=cal.to_dict(),
        new=metrics(rows,[CAUSES[j] for j in probabilities.argmax(1)]),
        old=metrics(rows,[CAUSES[j] for j in old.argmax(1)]),records=outrows,provenance=provenance(args))
    # Covariate variants are shared by bodies i % n_variants. A grouped bootstrap
    # complements the requested per-fall Wilson intervals, especially for BA.
    rng=np.random.default_rng(80500)
    variants=np.array([r['body']%data['config']['n_variants'] for r in rows])
    unique=np.unique(variants)
    labels=np.array([r['label'] for r in rows])
    for name,post in (('new',probabilities),('old',old)):
        guessed=np.array([CAUSES[j] for j in post.argmax(1)])
        balanced=[]
        accuracy=[]
        for _ in range(5000):
            draws=rng.choice(unique,len(unique),replace=True)
            counts=np.bincount(draws,minlength=int(unique.max())+1)[variants]
            rec=[]
            for cause in CAUSES[:3]:
                mask=labels==cause
                if counts[mask].sum():
                    rec.append(float(np.sum(counts[mask]*(guessed[mask]==cause))/counts[mask].sum()))
            # every cause present in the data (the speed control has no gait falls)
            if len(rec)==len(set(labels)):
                balanced.append(float(np.mean(rec)))
                accuracy.append(float(np.sum(counts*(guessed==labels))/counts.sum()))
        result[name]['balanced_accuracy_cluster_bootstrap95']=np.quantile(balanced,[.025,.975]).tolist()
        result[name]['accuracy_cluster_bootstrap95']=np.quantile(accuracy,[.025,.975]).tolist()
    result['bootstrap']=dict(seed=80500,replicates=5000,unit='shared randomized dynamics variant')
    result['solver_warnings']=data.get('solver_warnings',[])
    result['pre_injection_falls']=data['pre_injection_falls']
    result['injected_survivors']=data['injected_survivors']
    result['post_injection_nonfalls']=data['post_injection_nonfalls']
    save(args.output,result)
    print(json.dumps({k:result[k] for k in ('seed','calibration_seed','new','old')},indent=2))


def audit(args):
    args.input=Path(args.input)
    data=json.loads(args.input.read_text(encoding='utf-8'))
    gate=json.loads(args.gate.read_text(encoding='utf-8'))
    assert data['complete'] and len(data['records'])==len(gate['records'])
    assert len({r['body'] for r in gate['records']})==len(gate['records'])
    legacy_errors=[]
    for r in data['records']:
        trace=unpack_trace(r['evidence']).astype(float)
        tail=trace[-100:]
        cs=np.vstack((np.zeros((1,2)),np.cumsum(tail[:,:2],axis=0)))
        u=float(np.linalg.norm(cs[10:]-cs[:-10],axis=1).max())
        speed=tail[:,13]
        slip=float(np.min(np.stack([speed[k:len(speed)-2+k] for k in range(3)]),axis=0).max())
        error=float(np.max(np.abs(np.array([u,slip])-r['legacy_features'])))
        assert error<1e-4
        legacy_errors.append(error)
    rows=gate['records']
    new=np.array([r['inferred']==r['truth'] for r in rows])
    old=np.array([r['old_inferred']==r['truth'] for r in rows])
    groups=np.array([r['body']%data['config']['n_variants'] for r in rows])
    ids=np.unique(groups)
    rng=np.random.default_rng(80501)
    deltas=[]
    for _ in range(5000):
        drawn=rng.choice(ids,len(ids),replace=True)
        weights=np.bincount(drawn,minlength=int(ids.max())+1)[groups]
        deltas.append(float(np.sum(weights*(new.astype(float)-old.astype(float)))/weights.sum()))
    late=[r for r in rows if r['truth']=='push' and r['fall_time']-r['truth_event']['time']>2.]
    delay={cause:[r['fall_time']-r['truth_event']['time'] for r in rows if r['truth']==cause]
           for cause in ('gait','push','slip')}
    out=dict(rows_verified=len(rows),legacy_U_S_max_reconstruction_error=max(legacy_errors),
        new_only_correct=int(np.sum(new&~old)),old_only_correct=int(np.sum(old&~new)),
        accuracy_gain=float(new.mean()-old.mean()),paired_variant_bootstrap95=np.quantile(deltas,[.025,.975]).tolist(),
        bootstrap=dict(seed=80501,replicates=5000,unit='shared randomized dynamics variant'),
        late_push_falls=len(late),late_push_new_correct=sum(r['inferred']=='push' for r in late),
        late_push_old_correct=sum(r['old_inferred']=='push' for r in late),
        delay_seconds={c:dict(minimum=min(v),median=float(np.median(v)),maximum=max(v)) for c,v in delay.items()},
        model_hashes_unchanged={k:hashlib.sha256(Path(path).read_bytes()).hexdigest()==data['provenance'][k]
          for k,path in [('policy_sha256','runs/locomotion/policy_v3/policy.pt'),
                         ('body_model_checkpoint_sha256','runs/locomotion/policy_v3/ckpt_last.pt')]},
        source_sha256=dict(data=hashlib.sha256(args.input.read_bytes()).hexdigest(),gate=hashlib.sha256(args.gate.read_bytes()).hexdigest()),
        command=[sys.executable,*sys.argv])
    assert all(out['model_hashes_unchanged'].values())
    args.output.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=('collect','fit','score','audit'))
    ap.add_argument('--config',default='configs/locomotion_v3c.yaml')
    ap.add_argument('--policy',default='runs/locomotion/policy_v3/policy.pt')
    ap.add_argument('--ckpt',default='runs/locomotion/policy_v3/ckpt_last.pt')
    ap.add_argument('--device',default='cuda',choices=('cuda','cpu'))
    ap.add_argument('--seed',type=int)
    ap.add_argument('--per',type=int,default=60)
    ap.add_argument('--threads',type=int,default=6)
    ap.add_argument('--chunk',type=int,default=48)
    ap.add_argument('--conditions',default='gait,push,slip')
    ap.add_argument('--inject-at',type=float,default=3.)
    ap.add_argument('--after',type=float,default=4.)
    ap.add_argument('--push-command',type=float,default=None,
                    help='forward goal (m/s) set at the push; default keeps the 0.6 m/s walk')
    ap.add_argument('--slip-command',type=float,default=1.2,
                    help='forward goal (m/s) set when the floor turns to ice (original protocol: 1.2)')
    ap.add_argument('--inputs',nargs='+')
    ap.add_argument('--input')
    ap.add_argument('--model')
    ap.add_argument('--calibration')
    ap.add_argument('--gate',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve all prior data; choose a new path')
    if args.mode=='collect': collect(args)
    elif args.mode=='fit': fit(args)
    elif args.mode=='score': score(args)
    else: audit(args)


if __name__=='__main__':
    main()
