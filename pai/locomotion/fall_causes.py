"""Sensor-only fall evidence and fitted Bayesian likelihoods.

No injection metadata, friction coefficients, externally applied forces, body
indices or event logs are inputs to feature extraction or inference.
"""
import base64
import math
import zlib
import numpy as np
import torch
from pai.locomotion.env import PROPRIO
from pai.locomotion.falls import SurpriseMonitor

CAUSES=('gait','push','slip','unknown')
CHANNELS=('res_vx','res_vy','res_vz','res_angular','res_gravity','res_joint_pos',
          'res_joint_vel','surprise','loaded_slide','shear_ratio','load_sum',
          'load_imbalance','loaded_feet','foot_speed','torque_ratio','upright')


def instrumented_body():
    """Add read-only ankle load-cell sensors; change no physical parameter."""
    import mujoco
    from pai.locomotion.body import build_spec,G1Body,FEET
    spec=build_spec()
    for foot in FEET:
        for kind in ('force','torque'):
            spec.add_sensor(name=f'{foot}_fall_{kind}',type=getattr(mujoco.mjtSensor,'mjSENS_'+kind.upper()),
                            objtype=mujoco.mjtObj.mjOBJ_SITE,objname=foot)
    body=G1Body(spec.compile())
    addresses={kind:[int(body.model.sensor(f'{f}_fall_{kind}').adr[0]) for f in FEET]
               for kind in ('force','torque')}
    return body,addresses


def load_cells(sensors,addresses):
    return tuple(np.stack([sensors[:,a:a+3] for a in addresses[kind]],axis=1) for kind in ('force','torque'))


class FallEvidenceMonitor:
    def __init__(self,model,n,device='cpu',chunk=48,contact_height=.03):
        self.model,self.n,self.device,self.chunk=model,n,device,chunk
        self.contact_height=contact_height
        self.history=[[] for _ in range(n)]
        self.legacy=SurpriseMonitor(None,n,window=100,contact_height=contact_height)

    def reset(self,ids):
        for i in np.atleast_1d(ids):
            self.history[int(i)]=[]
        self.legacy.reset(ids)

    @torch.no_grad()
    def update(self,obs,action,next_obs,sole_height,sole_vel,force,torque,active=None):
        residuals=[]
        surprises=[]
        for start in range(0,self.n,self.chunk):
            s=torch.as_tensor(obs[start:start+self.chunk,PROPRIO],dtype=torch.float32,device=self.device)
            a=torch.as_tensor(action[start:start+self.chunk],dtype=torch.float32,device=self.device)
            s2=torch.as_tensor(next_obs[start:start+self.chunk,PROPRIO],dtype=torch.float32,device=self.device)
            pred=self.model.predict(s,a)
            residuals.append((s2-pred).cpu().numpy())
            sur,_=self.model.surprise_and_disagreement(s,a,s2)
            surprises.append(sur.cpu().numpy())
        res=np.concatenate(residuals)
        sur=np.concatenate(surprises)
        std=self.model.out_norm.std.detach().cpu().numpy()
        normalized=res/np.maximum(std,1e-3)
        speed=np.linalg.norm(sole_vel[:,:,:2],axis=2)
        down=sole_height<self.contact_height
        normal=np.abs(force[:,:,2])
        loaded=down & (normal>20.)
        load=normal*loaded
        total=load.sum(1)
        slide=(speed*load).sum(1)/np.maximum(total,1.)
        shear=(np.linalg.norm(force[:,:,:2],axis=2)*loaded).sum(1)/np.maximum(total,20.)
        imbalance=np.abs(load[:,0]-load[:,1])/np.maximum(total,1.)
        tratio=(np.linalg.norm(torque,axis=2)*loaded).sum(1)/np.maximum(total,20.)
        values=np.column_stack((res[:,6:9],np.linalg.norm(normalized[:,:3],axis=1),
            np.linalg.norm(normalized[:,3:6],axis=1),np.linalg.norm(normalized[:,9:21],axis=1),
            np.linalg.norm(normalized[:,21:33],axis=1),sur,slide,shear,total,imbalance,
            loaded.sum(1),np.max(speed*down,axis=1),tratio,-next_obs[:,5]))
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite sensor evidence')
        for i in np.flatnonzero(np.ones(self.n,bool) if active is None else active):
            self.history[i].append(values[i].astype(np.float32))
        # Reuse exactly the old U/S calculation on the SAME transitions.
        k=self.legacy.k%self.legacy.window
        self.legacy.res[:,k]=res[:,6:8]
        self.legacy.slide[:,k]=np.max(speed*down,axis=1)
        self.legacy.sur[:,k]=sur
        self.legacy.k+=1


def _impulse(trace,width):
    if len(trace)<width:
        return 0.
    c=np.vstack((np.zeros((1,2)),np.cumsum(trace[:,:2],axis=0)))
    return float(np.linalg.norm(c[width:]-c[:-width],axis=1).max())


def extract_features(trace,dt=.02):
    """Search the full first-episode history, with no known intervention time.

    Both whole-history and pre-collapse summaries are retained: blindly removing
    the terminal interval also removes evidence of pushes causing immediate falls.
    """
    x=np.asarray(trace,dtype=float)
    if x.ndim!=2 or x.shape[1]!=len(CHANNELS) or not np.isfinite(x).all() or len(x)<15:
        raise ValueError('At least 15 finite sensor transitions required')
    # Ignore reset transients, using a fixed sensor warm-up, not the injection.
    warmup=min(15,max(0,len(x)-15))
    x=x[warmup:]
    prefix=x[:-10] if len(x)>25 else x
    baseline=x[:min(50,max(5,len(x)//3))]
    reference=max(float(np.median(baseline[:,10])),20.)
    feature={}
    for name,part in (('all',x),('precollapse',prefix)):
        for width in (5,10,25):
            feature[f'{name}_impulse_{width}']=_impulse(part,width)
        for j,label in ((2,'vertical_residual'),(3,'angular_residual'),(4,'gravity_residual'),
                        (5,'joint_position_residual'),(6,'joint_velocity_residual'),(7,'surprise'),
                        (8,'loaded_sliding'),(9,'shear_ratio'),(11,'load_imbalance'),(14,'torque_ratio')):
            v=np.abs(part[:,j])
            feature[f'{name}_{label}_median']=float(np.median(v))
            feature[f'{name}_{label}_p90']=float(np.quantile(v,.9))
        feature[name+'_load_p10']=float(np.quantile(part[:,10]/reference,.1))
        feature[name+'_load_p90']=float(np.quantile(part[:,10]/reference,.9))
        feature[name+'_no_support']=float(np.mean(part[:,12]==0))
        feature[name+'_double_support']=float(np.mean(part[:,12]==2))
        feature[name+'_loaded_slip_fraction']=float(np.mean(part[:,8]>.3))
    # Unknown-onset searches: strongest before/after contrast, not event time.
    onsets={}
    for j,label in ((0,'horizontal_residual'),(8,'loaded_sliding'),(9,'shear_ratio'),(10,'load')):
        signal=np.linalg.norm(prefix[:,:2],axis=1) if j==0 else prefix[:,j]/(reference if j==10 else 1.)
        contrasts=[abs(float(signal[t:t+5].mean()-signal[t-10:t].mean())) for t in range(10,len(signal)-5)]
        if contrasts:
            peak=int(np.argmax(contrasts))+10
            feature[label+'_change']=max(contrasts)
            feature[label+'_onset_age']=float((len(x)-peak)*dt)
            onsets[label]=dict(step=peak+warmup,contrast=max(contrasts))
        else:
            feature[label+'_change']=0.
            feature[label+'_onset_age']=0.
    return feature,onsets


def pack_trace(trace):
    x=np.asarray(trace,dtype='<f4')
    return dict(shape=list(x.shape),dtype='<f4',zlib_base64=base64.b64encode(zlib.compress(x.tobytes())).decode('ascii'))


def unpack_trace(record):
    data=zlib.decompress(base64.b64decode(record['zlib_base64']))
    return np.frombuffer(data,dtype=record['dtype']).reshape(record['shape']).copy()


class LearnedFallLikelihood:
    def __init__(self,names,center,scale,means,covariances,unknown_scale=4.,unknown_prior=.03):
        self.names=list(names)
        self.center=np.asarray(center)
        self.scale=np.asarray(scale)
        self.means=np.asarray(means)
        self.covariances=np.asarray(covariances)
        self.unknown_scale,self.unknown_prior=unknown_scale,unknown_prior

    @classmethod
    def fit(cls,features,labels,shrinkage=.5,unknown_scale=4.,unknown_prior=.03):
        names=sorted(features[0])
        x=np.log1p([[f[k] for k in names] for f in features])
        center=x.mean(0)
        scale=np.maximum(x.std(0),.05)
        z=(x-center)/scale
        means=[]
        cov=[]
        for cause in CAUSES[:3]:
            part=z[np.asarray(labels)==cause]
            if len(part)<3:
                raise ValueError('At least three development falls per cause')
            means.append(part.mean(0))
            c=np.atleast_2d(np.cov(part,rowvar=False))
            c=(1-shrinkage)*c+shrinkage*np.diag(np.maximum(np.diag(c),.1))+.03*np.eye(len(names))
            cov.append(c)
        return cls(names,center,scale,means,cov,unknown_scale,unknown_prior)

    def posterior(self,features):
        x=np.log1p(np.asarray([[f[k] for k in self.names] for f in features]))
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite feature values')
        z=(x-self.center)/self.scale
        d=z.shape[1]
        likelihood=[]
        for mean,cov in zip(self.means,self.covariances):
            diff=z-mean
            mahal=np.sum(diff*np.linalg.solve(cov,diff.T).T,axis=1)
            likelihood.append(-.5*(mahal+np.linalg.slogdet(cov)[1]+d*np.log(2*np.pi))+np.log((1-self.unknown_prior)/3))
        # Broad Student-t catch-all. Form/prior given; center/scale learned from
        # pooled development evidence, with no labelled unknown examples.
        nu=3.
        q=np.sum(z*z,axis=1)/(self.unknown_scale**2)
        unknown=(math.lgamma((nu+d)/2)-math.lgamma(nu/2)-d/2*np.log(nu*np.pi)
                 -d*np.log(self.unknown_scale)-(nu+d)/2*np.log1p(q/nu)+np.log(self.unknown_prior))
        ll=np.column_stack([*likelihood,unknown])
        ll-=ll.max(1,keepdims=True)
        p=np.exp(ll)
        return p/p.sum(1,keepdims=True)

    def to_dict(self):
        return dict(names=self.names,center=self.center.tolist(),scale=self.scale.tolist(),
                    means=self.means.tolist(),covariances=self.covariances.tolist(),
                    unknown_scale=self.unknown_scale,unknown_prior=self.unknown_prior)
