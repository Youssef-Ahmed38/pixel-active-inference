"""Scoring utilities: events/labels are never imported by predictive training."""
import json,math
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

CAUSES=['none','push','heavier_object','slippery_object','camera_shift']

def truth_rows(split):
    rows=[]
    for path in sorted(Path('results').glob(f'thinker_hier_{split}_truth*.json')):
        rows+=json.loads(path.read_text(encoding='utf8'))['episodes']
    return {r['index']:r for r in rows}

def cause(row):
    events=row['disturbances']
    if not events:return 'none'
    name=events[0]['disturbance']
    return {'payload':'heavier_object','friction':'slippery_object'}.get(name,name)

def rate(k,n):
    if not n:return dict(k=0,n=0,rate=None,wilson95=None)
    z=1.95996398454;p=k/n;den=1+z*z/n
    mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return dict(k=int(k),n=int(n),rate=p,wilson95=[max(0.,mid-half),min(1.,mid+half)])

def tau(state):
    x=np.asarray(state,float);x=x-x.mean(0)
    if np.square(x).sum()<1e-12:return dict(steps=None,censored=False)
    cap=min(40,len(x)//2)
    for lag in range(1,cap+1):
        a,b=x[:-lag],x[lag:];den=np.sqrt(np.square(a).sum()*np.square(b).sum())
        rho=(a*b).sum()/den if den>1e-12 else 0.
        if rho<=np.exp(-1):return dict(steps=lag,censored=False)
    return dict(steps=cap,censored=True)

def run_streams(model,streams,norm,device='cuda',refine=None,batch=16):
    from pai.thinker.hier_data import normalized
    results=[];model.eval()
    for lo in range(0,len(streams),batch):
        selected=[normalized(s,norm) for s in streams[lo:lo+batch]];lengths=[len(s['action']) for s in selected];longest=max(lengths)
        tensors=[]
        for key,extra in [('state',1),('action',0),('context',0)]:
            padded=[]
            for r,n in zip(selected,lengths):
                ids=np.minimum(np.arange(longest+extra),n if key=='state' else n-1)
                padded.append(r[key][ids])
            tensors.append(torch.as_tensor(np.stack(padded),dtype=torch.float32,device=device))
        s,a,m=tensors
        with torch.no_grad():prediction,traces=model(s[:,:-1],a,m,refine=refine)
        error=(prediction-s[:,1:]).square().mean(-1).cpu().numpy()
        tr={k:v.detach().cpu().numpy() for k,v in traces.items()}
        for i,n in enumerate(lengths):
            state=tr['level2'][i,:n]
            results.append(dict(embedding=np.r_[state.mean(0),state.std(0)],error=error[i,:n],
              code_probabilities=state[:,:model.cfg.codes] if hasattr(model,'latent') else None,
              traces={k:v[i,:n] for k,v in tr.items()}))
    return results

def fit_readouts(training,development,labels,seed):
    scaler=StandardScaler().fit(np.stack([r['embedding'] for r in training]))
    train=scaler.transform(np.stack([r['embedding'] for r in training]))
    pca=PCA(n_components=16,whiten=True,random_state=seed).fit(train)
    tr=pca.transform(train)
    dv=pca.transform(scaler.transform(np.stack([r['embedding'] for r in development])))
    clusters=KMeans(n_clusters=5,n_init=10,random_state=seed).fit(tr)
    pred=clusters.predict(dv);matrix=np.zeros((5,5),int)
    for k,y in zip(pred,labels):matrix[k,y]+=1
    a,b=linear_sum_assignment(-matrix);mapping={int(x):int(y) for x,y in zip(a,b)}
    probe=LogisticRegression(C=1.,max_iter=1000,random_state=seed).fit(dv,labels)
    return dict(scaler=scaler,pca=pca,clusters=clusters,mapping=mapping,probe=probe,development_confusion=matrix.tolist())

def predict_readouts(readouts,results):
    x=readouts['pca'].transform(readouts['scaler'].transform(np.stack([r['embedding'] for r in results])))
    cluster=readouts['clusters'].predict(x)
    return np.array([readouts['mapping'][int(k)] for k in cluster]),readouts['probe'].predict(x),cluster

def episode_metrics(result,truth,clean_level):
    error=result['error'];n=len(error);events=truth['disturbances']
    onset=min((max(0,(int(e['t'])-1)//5) for e in events),default=None)
    onset=onset if onset is not None and onset<n else None
    post=error[onset:min(n,onset+20)] if onset is not None else np.array([])
    adaptation=None;initial=False
    if len(post):
        initial=bool(post[:min(3,len(post))].mean()>clean_level)
        for t in range(max(0,len(post)-2)):
            if post[t:t+3].mean()<=clean_level:adaptation=t+2;break
    traces=result['traces'];its=traces['iterations'];active=its>0
    improved=traces['refinement_before'][active]-traces['refinement_after'][active]
    probs=result.get('code_probabilities')
    return dict(code_index=int(probs.mean(0).argmax()) if probs is not None else None,
        code_entropy=float(-(probs*np.log(np.maximum(probs,1e-12))).sum(1).mean()) if probs is not None else None,
        mean_code_probabilities=probs.mean(0).tolist() if probs is not None else None,
        mse=float(error.mean()),onset=onset,complete_post20=len(post)==20,
        post_available_steps=len(post),post_available_mse=float(post.mean()) if len(post) else None,
        post20_mse=float(post.mean()) if len(post)==20 else None,
        adaptation_steps=adaptation,censored=bool(len(post) and adaptation is None),initial_spike=initial,
        restricted_adaptation=(min(20,adaptation) if adaptation is not None else 20) if len(post)==20 else None,
        early_error=float(post[:5].mean()) if len(post) else None,
        late_error=float(post[15:20].mean()) if len(post)>=20 else None,
        error_curve=[float(v) for v in error],post_curve=[float(v) for v in post],
        mean_iterations=float(its.mean()),update_iterations=float(its[active].mean()) if active.any() else 0.,
        inner_past_error_improvement=float(improved.mean()) if len(improved) else 0.,
        iterations=its.tolist(),surprise=traces['surprise'].tolist(),
        timescales={k:tau(traces[k]) for k in ('level1','level2','level3')})

def valid_mean(values):
    values=[v for v in values if v is not None]
    return float(np.mean(values)) if values else None

def summarize(records):
    labels=[r['label'] for r in records];cluster=[r['cluster_cause'] for r in records];probe=[r['probe_cause'] for r in records]
    def mean(key):
        vals=[r[key] for r in records if r[key] is not None]
        return float(np.mean(vals)) if vals else None
    return dict(code_usage={str(k):sum(r['code_index']==k for r in records) for k in range(8)} if records[0]['code_index'] is not None else None,
        code_entropy=mean('code_entropy'),cluster_accuracy=rate(sum(a==b for a,b in zip(labels,cluster)),len(records)),
        probe_accuracy=rate(sum(a==b for a,b in zip(labels,probe)),len(records)),
        nmi=float(normalized_mutual_info_score(labels,[r['cluster_index'] for r in records])),
        mse=mean('mse'),post20_mse=mean('post20_mse'),restricted_adaptation=mean('restricted_adaptation'),
        complete_post20=sum(r['complete_post20'] for r in records),
        truncated_post20=sum(r['onset'] is not None and not r['complete_post20'] for r in records),
        initial_spikes=sum(r['initial_spike'] for r in records if r['complete_post20']),
        censored_adaptations=sum(r['censored'] for r in records if r['complete_post20']),
        mean_iterations=mean('mean_iterations'),update_iterations=mean('update_iterations'),
        inner_past_error_improvement=mean('inner_past_error_improvement'),
        timescales={k:dict(mean=valid_mean([r['timescales'][k]['steps'] for r in records]),
                          censored=sum(r['timescales'][k]['censored'] for r in records),
                          undefined=sum(r['timescales'][k]['steps'] is None for r in records)) for k in ('level1','level2','level3')},
        cluster_by_condition={c:rate(sum(r['label']==r['cluster_cause'] for r in records if r['label']==c),sum(r['label']==c for r in records)) for c in CAUSES})
