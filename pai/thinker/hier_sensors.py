"""RGB-only object measurements, given calibrated geometry/color decoder; own arm sensors."""
import numpy as np
from pai.perception.slot_tokens import PixelGeometry
from pai.world.entities import POS,VEL,YAW,GRIP,KIND,COLOR,FORCE,TOKEN_DIM,KINDS

class OwnPixelPerception:
    allow_recalibration=False
    def __init__(self,env):
        self.env=env
        camera=env.cfg.cameras[0].name
        self.geom=PixelGeometry(env,camera,env.cfg.image_size)
        self.specs=list(env.objects.values())
        self.camera=camera
        rgba={b.name:b.rgba for b in env.cfg.blocks}
        self.colours=[rgba[s.name] if s.name in rgba else env.cfg[s.kind].rgba for s in self.specs]
        self.reset()
    def reset(self):
        self.previous=None;self.previous_ee=None;self.held=None
        self.features=[];self.images=[]
    def __call__(self,obs,prev_ee=None):
        # Deliberate allowlist: never touch obs['state'], predicates, env.data,
        # current camera pose, true object pose, event logs, or disturbance fields.
        image=self.env.render(self.camera)
        self.geom.label(image)
        self.features.append(np.r_[image.mean((0,1)),image.std((0,1))].astype(np.float32)/255)
        if len(self.images)<1:self.images.append(image.copy())
        ee=np.asarray(obs['ee_pos']);grip=float(obs['gripper'])
        n=len(self.specs);positions=np.zeros((n,3),np.float32)
        for i,s in enumerate(self.specs):
            mask,_=self.geom.masks(i)
            v,u=np.where(mask)
            prior=(self.previous[i] if self.previous is not None else np.array([.5,0,self.geom.rest_z[i]]))
            if len(u)>8:
                z=float(prior[2]) if self.held==i else self.geom.rest_z[i]
                estimate=self.geom.on_plane(float(u.mean()),float(v.mean()),z)
                positions[i]=np.clip(estimate,[.15,-.5,0],[.9,.5,.6])
            else:positions[i]=prior
        blocked=.005<grip<.065
        if not blocked:self.held=None
        elif self.held is None:
            distance=np.linalg.norm(positions-ee,axis=1)
            j=int(distance.argmin())
            if self.specs[j].kind=='block' and distance[j]<.07:self.held=j
        if self.held is not None:
            j=self.held
            # Touch supplies grasp xy; visible silhouette supplies object z.
            start=np.r_[ee[:2],ee[2]-.015]
            estimate=self.geom.fit(j,start,free='z',steps=(.008,.004,.002))
            positions[j]=np.r_[ee[:2],np.clip(estimate[2],self.geom.rest_z[j],.6)]
        out=np.zeros((n+1,TOKEN_DIM),np.float32)
        out[0,POS]=ee;out[0,YAW]=[0,1];out[0,GRIP]=grip
        out[0,FORCE]=obs['wrist_force'];out[0,KIND]=[1,0,0,0]
        if prev_ee is not None:out[0,VEL]=(ee-prev_ee)/self.env.control_dt
        for i,s in enumerate(self.specs):
            out[i+1,POS]=positions[i];out[i+1,YAW]=[0,1]
            out[i+1,KIND]=np.eye(4)[KINDS.index(s.kind)];out[i+1,COLOR]=self.colours[i][:3]
            if self.previous is not None:out[i+1,VEL]=(positions[i]-self.previous[i])/.1
        self.previous=positions.copy()
        return out


def sensor_vector(tokens,rgb,target=1):
    others=[i for i in range(1,len(tokens)) if i!=target]
    return np.r_[tokens[0,POS],tokens[0,FORCE],tokens[target,POS],tokens[others,POS].mean(0),
                 tokens[0,GRIP],rgb].astype(np.float32)
