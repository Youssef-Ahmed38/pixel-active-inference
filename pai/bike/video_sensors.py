"""Calibrated pixel-only motion reconstruction and an actual on-bike camera.

Geometry, marker identities and calibration are given. No expert state/control
labels enter the visual decoder. Proprioception exposes joint encoders and gyro,
never root pose, root velocity, tracking coordinates or simulator attitude.
"""
from dataclasses import dataclass
import math
import mujoco
import numpy as np
from pai.bike.env import BicycleEnv,XML

PALETTE=np.array([[1,0,1],[0,1,1],[1,1,0],[0,1,0],[.5,.25,1]],float)
FRAME_POINTS=np.array([[-.45,-.4,.25],[.45,-.4,.25],[-.45,.4,.25]])
FORK_POINT=np.array([0,.4,.4]);TORSO_POINT=np.array([0,.35,.5])
EYES=np.array([[5,-8,4],[5,8,4]],float);TARGET=np.array([5,0,.8])
FOV=75.


def camera_rotation(eye,target):
    forward=np.asarray(target)-eye;forward=forward/np.linalg.norm(forward)
    right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
    up=np.cross(right,forward)
    return np.stack((right,up,-forward),axis=1)


def video_xml():
    assets='<asset>'+''.join(f'<material name="tag{i}" rgba="{c[0]} {c[1]} {c[2]} 1" emission="1" specular="0"/>' for i,c in enumerate(PALETTE))
    assets+='<material name="road" rgba="1 1 1 1" emission="1" specular="0"/></asset>'
    cameras=''
    for i,eye in enumerate(EYES):
        r=camera_rotation(eye,TARGET)
        cameras+=f'<camera name="video{i}" pos="{" ".join(map(str,eye))}" xyaxes="{" ".join(map(str,np.r_[r[:,0],r[:,1]]))}" fovy="{FOV}"/>'
    road=''.join(f'<geom name="road{side}" type="box" pos="0 {side} .002" size="100 .025 .002" material="road" contype="0" conaffinity="0"/>' for side in (-1.5,1.5))
    xml=XML.replace('<worldbody>','<visual><global offwidth="512" offheight="512"/></visual>'+assets+'<worldbody>'+cameras+road)
    frame=''.join(f'<site name="tag{i}" pos="{" ".join(map(str,p))}" size=".07" material="tag{i}"/>' for i,p in enumerate(FRAME_POINTS))
    frame+='<camera name="onboard" pos=".3 0 .55" xyaxes="0 -1 0 0 0 1" fovy="75"/>'
    xml=xml.replace('<freejoint name="root"/>','<freejoint name="root"/>'+frame)
    xml=xml.replace('<joint name="steering"',f'<site name="tag3" pos="{" ".join(map(str,FORK_POINT))}" size=".07" material="tag3"/><joint name="steering"')
    xml=xml.replace('<joint name="torso_lean"',f'<site name="tag4" pos="{" ".join(map(str,TORSO_POINT))}" size=".07" material="tag4"/><joint name="torso_lean"')
    return xml


class VideoBicycleEnv(BicycleEnv):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.model=mujoco.MjModel.from_xml_string(video_xml());self.data=mujoco.MjData(self.model)
        self.joints={n:self.model.joint(n) for n in ('rear_spin','front_spin','steering','torso_lean')}
        self.reset()


def marker_centres(image):
    """Subpixel RGB centroids, with explicit missing-marker abstention."""
    rgb=np.asarray(image,dtype=np.float32)/255
    strength=rgb.max(2)-rgb.min(2)
    chroma=(rgb-rgb.min(2)[...,None])/np.maximum(strength[...,None],1e-8)
    points=[]
    xx=np.arange(image.shape[1]);yy=np.arange(image.shape[0])
    for colour in PALETTE:
        colour=(colour-colour.min())/(colour.max()-colour.min())
        distance=sum((chroma[:,:,i]-colour[i])**2 for i in range(3))
        weight=np.where((distance<.12**2)&(strength>.02),strength,0)
        if weight.sum()<.1:points.append([np.nan,np.nan])
        else:points.append([float(xx@weight.sum(0)/weight.sum()),float(yy@weight.sum(1)/weight.sum())])
    return np.asarray(points)


def projection(index,size):
    f=size/(2*np.tan(np.deg2rad(FOV/2)));eye=EYES[index];r=camera_rotation(eye,TARGET)
    # Pixel rows point down; camera forward is local -z.
    rot=np.stack((r[:,0],-r[:,1],-r[:,2]))
    k=np.array([[f,0,(size-1)/2],[0,f,(size-1)/2],[0,0,1]])
    return k@np.column_stack((rot,-rot@eye))


def triangulate(left,right,size):
    matrices=[projection(0,size),projection(1,size)];points=[]
    for a,b in zip(left,right):
        if not np.isfinite([a,b]).all():points.append([np.nan]*3);continue
        equations=np.stack([a[0]*matrices[0][2]-matrices[0][0],a[1]*matrices[0][2]-matrices[0][1],
                            b[0]*matrices[1][2]-matrices[1][0],b[1]*matrices[1][2]-matrices[1][1]])
        _,_,v=np.linalg.svd(equations);points.append(v[-1,:3]/v[-1,3])
    return np.asarray(points)


def rotation_angle(vector,reference,axis):
    axis=np.asarray(axis,float);axis/=np.linalg.norm(axis)
    a=reference-axis*(reference@axis);b=vector-axis*(vector@axis)
    return float(np.arctan2(axis@np.cross(a,b),a@b))


def reconstruct_pose(left_image,right_image):
    pts=triangulate(marker_centres(left_image),marker_centres(right_image),left_image.shape[0])
    if not np.isfinite(pts).all():return None
    origin=FRAME_POINTS.mean(0);observed=pts[:3].mean(0)
    u,_,vt=np.linalg.svd((FRAME_POINTS-origin).T@(pts[:3]-observed))
    r=vt.T@u.T
    if np.linalg.det(r)<0:vt[-1]*=-1;r=vt.T@u.T
    root=observed-r@origin
    quaternion=np.empty(4);mujoco.mju_mat2Quat(quaternion,r.ravel())
    fork=r.T@(pts[3]-root)-np.array([.5,0,-.1])
    torso=r.T@(pts[4]-root)-np.array([-.05,0,.1])
    steer=rotation_angle(fork,FORK_POINT,[-.3,0,.953939])
    lean=rotation_angle(torso,TORSO_POINT,[1,0,0])
    return np.r_[root,quaternion,steer,lean]


@dataclass(frozen=True)
class OwnSenses:
    wheel_speed:float
    steering:float
    torso_lean:float
    gyro:tuple


def proprioception(env):
    return OwnSenses(float(env.data.qvel[env.joints['rear_spin'].dofadr[0]])*.32,
                     float(env.data.qpos[env.joints['steering'].qposadr[0]]),
                     float(env.data.qpos[env.joints['torso_lean'].qposadr[0]]),
                     tuple(map(float,env.data.sensor('angular_velocity').data)))


class OnboardVision:
    """Given geometric horizon/road decoder, aided by the rider's gyro only."""
    def __init__(self):self.roll=None;self.yaw=None

    def features(self,image,senses,dt):
        if image is None:
            if self.roll is None:raise ValueError('initial on-board image required')
            self.roll+=senses.gyro[0]*dt;self.yaw+=senses.gyro[2]*dt
            return np.array([self.roll,senses.gyro[0],senses.wheel_speed,senses.steering,senses.torso_lean,self.yaw])
        image=np.asarray(image);h,w=image.shape[:2];grey=image.mean(2)
        # Infinite ground/black sky boundary. Ignore the bottom, where the bike lives.
        columns=np.arange(4,w-4);edges=[]
        for x in columns:
            values=np.flatnonzero(grey[:int(.8*h),x]>5)
            edges.append(values[0] if len(values) else np.nan)
        edges=np.asarray(edges);good=np.isfinite(edges)&(edges>.15*h)&(edges<.75*h)
        visible=False
        if good.sum()>w//3:
            slope,intercept=np.polyfit(columns[good],edges[good],1)
            measured=-math.atan(slope);visible=True
        else:measured=0.
        predicted=measured if self.roll is None else self.roll+senses.gyro[0]*dt
        self.roll=predicted+.1*(measured-predicted) if visible else predicted
        # White road edges converge to the forward vanishing point.
        white=(image.min(2)>225);ys,xs=np.where(white&(np.indices((h,w))[0]>.55*h))
        f=h/(2*np.tan(np.deg2rad(FOV/2)));yaw_est=None
        splits=[xs<w/2,xs>=w/2];lines=[]
        for mask in splits:
            if mask.sum()>8:lines.append(np.polyfit(ys[mask],xs[mask],1))
        if len(lines)==2 and abs(lines[0][0]-lines[1][0])>.05:
            vy=(lines[1][1]-lines[0][1])/(lines[0][0]-lines[1][0]);vx=lines[0][0]*vy+lines[0][1]
            if -.5*w<vx<1.5*w:yaw_est=math.atan((vx-(w-1)/2)/f)
        if self.yaw is None:self.yaw=yaw_est if yaw_est is not None else 0.
        else:self.yaw+=senses.gyro[2]*dt
        if yaw_est is not None:self.yaw+=.05*(yaw_est-self.yaw)
        return np.array([self.roll,senses.gyro[0],senses.wheel_speed,senses.steering,senses.torso_lean,self.yaw])
