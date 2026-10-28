"""Learn riding targets from calibrated RGB videos, never expert state labels."""
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
import mujoco
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from pai.bike.video_sensors import VideoBicycleEnv,reconstruct_pose,OnboardVision
from pai.bike.observation_learning import ObservedRidingPolicy,infer_action


@dataclass(frozen=True)
class PixelEpisode:
    rgb:np.ndarray             # [time, stereo view, height, width, RGB]
    times:np.ndarray

    def __post_init__(self):
        if self.rgb.ndim!=5 or self.rgb.shape[1]!=2 or self.rgb.shape[-1]!=3 or self.rgb.dtype!=np.uint8:
            raise ValueError('stereo uint8 RGB video required')
        if len(self.rgb)!=len(self.times) or len(self.times)<9 or not np.isfinite(self.times).all() or (np.diff(self.times)<=0).any():
            raise ValueError('matching increasing frame times required')


def decode_motion(episode,inverse='mujoco'):
    """Known inverse graphics + temporal smoothing; given rolling-wheel model."""
    poses=[reconstruct_pose(a,b) for a,b in episode.rgb]
    good=np.array([p is not None for p in poses])
    if good.sum()<9:raise ValueError('too few fully visible marker frames')
    values=np.stack([p if p is not None else np.full(9,np.nan) for p in poses])
    for j in range(9):values[:,j]=np.interp(episode.times,episode.times[good],values[good,j])
    angles=Rotation.from_quat(values[:,[4,5,6,3]]).as_euler('xyz')
    angles=np.unwrap(angles,axis=0)
    # Joint rest at the initial frame is given by the benchmark's mechanics;
    # no expert joint/control labels are consulted for this calibration.
    values[:,7:]-=values[0,7:]
    raw=np.column_stack((values[:,:3],angles,values[:,7:]))
    window=min(11,len(raw) if len(raw)%2 else len(raw)-1)
    smooth=savgol_filter(raw,window,3,axis=0)
    q=np.zeros((len(raw),11));q[:,:3]=smooth[:,:3]
    q[:,3:7]=Rotation.from_euler('xyz',smooth[:,3:6]).as_quat()[:,[3,0,1,2]]
    distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(smooth[:,:2],axis=0),axis=1))]
    q[:,7]=distance/.32;q[:,9]=distance/.32;q[:,8]=smooth[:,6];q[:,10]=smooth[:,7]
    env=VideoBicycleEnv();static=SimpleNamespace(model=env.model,joints=env.joints)
    vel=np.zeros((len(q),10))
    for i in range(len(q)):
        a=max(0,i-1);b=min(len(q)-1,i+1)
        mujoco.mj_differentiatePos(env.model,vel[i],episode.times[b]-episode.times[a],q[a],q[b])
    vel=savgol_filter(vel,window,3,axis=0)
    acc=np.gradient(vel,episode.times,axis=0);acc=savgol_filter(acc,window,3,axis=0)
    speed=vel[:,0]*np.cos(smooth[:,5])+vel[:,1]*np.sin(smooth[:,5])
    if inverse=='mujoco':
        actions=np.stack([infer_action(static,p,v,a) for p,v,a in zip(q,vel,acc)])
    elif inverse=='servo':
        # Given low-bandwidth approximation: rolling longitudinal inertia and
        # each servo's own inertia/gravity/damping. Unlike the full constrained
        # inverse, noisy visual root height cannot switch contact constraints.
        acceleration=np.gradient(speed,episode.times)
        actions=[];data=mujoco.MjData(env.model);mass=env.model.body_mass.sum()+2.
        for i in range(len(q)):
            data.qpos[:]=q[i];data.qpos[2]+=2.;data.qvel[:]=vel[i]
            mujoco.mj_forward(env.model,data)
            inertia=np.zeros((env.model.nv,env.model.nv));mujoco.mj_fullM(env.model,data,inertia)
            targets=[]
            for name,kp,kv in [('steering',100.,12.),('torso_lean',100.,15.)]:
                j=env.joints[name];d=int(j.dofadr[0]);position=int(j.qposadr[0])
                force=data.qfrc_bias[d]+env.model.dof_damping[d]*vel[i,d]+inertia[d,d]*acc[i,d]
                targets.append(q[i,position]+(force+kv*vel[i,d])/kp)
            actions.append([mass*.32*acceleration[i]+.03*speed[i]/.32,*targets])
        actions=np.clip(actions,env.model.actuator_ctrlrange[:,0],env.model.actuator_ctrlrange[:,1])
    else:raise ValueError('unknown video inverse')
    features=np.column_stack((smooth[:,3],vel[:,3],speed,q[:,8],q[:,10],smooth[:,5]))
    return features,actions,{'valid_marker_frames':int(good.sum()),'frames':len(q),
                            'estimated_qpos':q,'estimated_qvel':vel,
                            'observed_window_mask':np.convolve(good.astype(int),np.ones(window,dtype=int),'same')==window}


def fit_decoded_policy(decoded,ridge=.1,inverse='mujoco',feature_set='all'):
    rows=[];targets=[];counts=[];base_weights=[]
    for x,y,info in decoded:
        # Avoid polynomial edge extrapolation; no truth-based filtering.
        mask=info['observed_window_mask']
        if mask.sum()<5:raise ValueError('too few fully observed temporal windows')
        rows.append(x[mask]);targets.append(y[mask]);counts.append(info['valid_marker_frames'])
        base_weights.extend([1./mask.sum()]*int(mask.sum()))
    x=np.concatenate(rows);y=np.concatenate(targets)
    if feature_set not in ('all','no_joints','no_joints_constant_lean'):raise ValueError('unknown feature set')
    if feature_set!='all':x=x.copy();x[:,[3,4]]=0.
    base_weights=np.asarray(base_weights)*len(x)/len(rows)
    mean=np.average(x,axis=0,weights=base_weights)
    scale=np.maximum(np.sqrt(np.average((x-mean)**2,axis=0,weights=base_weights)),.01)
    z=np.column_stack(((x-mean)/scale,np.ones(len(x))))
    limits=np.array([12.,.6,.25]);norm=y/limits
    weights=base_weights.copy();coef=None
    for _ in range(5):
        coef=np.linalg.solve((z.T*weights)@z+ridge*np.eye(7),(z.T*weights)@norm)
        error=np.linalg.norm(z@coef-norm,axis=1)
        weights=base_weights*np.minimum(1.,.25/np.maximum(error,1e-6))
    if feature_set=='no_joints_constant_lean':
        coef[:,2]=0.;coef[6,2]=np.median(norm[:,2])
    physical=coef*limits;physical[:6]/=scale[:,None];physical[6]-=mean@physical[:6]
    return ObservedRidingPolicy(physical,'RGB stereo inverse graphics + given '+inverse+' inverse; expert states/controls unused'),{
        'episodes':len(rows),'decision_examples':len(x),'valid_marker_frames':counts,'ridge':ridge,
        'inverse':inverse,'feature_set':feature_set,'inferred_action_mean':y.mean(0).tolist(),'weights':physical.tolist()}


def fit_video_policy(episodes,ridge=.1,inverse='mujoco',feature_set='all'):
    return fit_decoded_policy((decode_motion(episode,inverse) for episode in episodes),ridge,inverse,feature_set)


class VideoRidingPolicy:
    def __init__(self,policy):self.policy=policy;self.vision=OnboardVision()
    def action(self,image,senses,dt):
        features=self.vision.features(image,senses,dt)
        return self.policy.action(features),features
