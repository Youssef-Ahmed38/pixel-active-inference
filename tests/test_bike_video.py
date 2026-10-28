"""Pixel teaching boundary, physical mounting, proprioception and learned policy."""
from types import SimpleNamespace
import numpy as np
import pytest
import mujoco
from pai.bike.env import BicycleEnv
from pai.bike.video_sensors import (VideoBicycleEnv,FRAME_POINTS,FORK_POINT,TORSO_POINT,
 projection,reconstruct_pose,OwnSenses,OnboardVision,proprioception)
from pai.bike.video_learning import PixelEpisode,VideoRidingPolicy,fit_video_policy
from pai.bike.observation_learning import ObservedRidingPolicy


def test_camera_is_rigidly_mounted_without_changing_dynamics():
    base=BicycleEnv();video=VideoBicycleEnv()
    camera=video.model.camera('onboard').id
    assert video.model.cam_bodyid[camera]==video.model.body('frame').id
    np.testing.assert_array_equal(video.model.body_mass,base.model.body_mass)
    np.testing.assert_array_equal(video.model.body_inertia,base.model.body_inertia)
    for env in (base,video):env.reset(5,speed=3.,roll=.03)
    for _ in range(5):
        for env in (base,video):env.data.ctrl[:]=[1.,.02,0];mujoco.mj_step(env.model,env.data)
    np.testing.assert_allclose(base.data.qpos,video.data.qpos,atol=1e-12)


def test_proprioception_cannot_read_root_pose_or_root_velocity():
    class Encoders:
        def __init__(self,allowed):self.allowed=allowed
        def __getitem__(self,index):
            assert isinstance(index,(int,np.integer)) and index in self.allowed
            return self.allowed[index]
    data=SimpleNamespace(qpos=Encoders({8:.02,10:-.01}),qvel=Encoders({6:10.}),
                         sensor=lambda name:SimpleNamespace(data=np.array([.1,.2,.3])) if name=='angular_velocity' else None)
    env=SimpleNamespace(data=data,joints={'rear_spin':SimpleNamespace(dofadr=[6]),
             'steering':SimpleNamespace(qposadr=[8]),'torso_lean':SimpleNamespace(qposadr=[10])})
    senses=proprioception(env)
    assert senses.wheel_speed==pytest.approx(3.2) and senses.gyro==(.1,.2,.3)


def test_pixel_episode_rejects_pose_tables_and_bad_timestamps():
    with pytest.raises(ValueError):PixelEpisode(np.zeros((10,11)),np.arange(10))
    with pytest.raises(ValueError):PixelEpisode(np.zeros((10,2,16,16,3),np.uint8),np.zeros(10))


def test_triangulation_uses_only_image_marker_measurements(monkeypatch):
    import pai.bike.video_sensors as module
    world=np.vstack((FRAME_POINTS+[5,0,.69],FORK_POINT+[5.5,0,.59],TORSO_POINT+[4.95,0,.79]))
    images=[]
    for cam in range(2):
        p=projection(cam,512)@np.column_stack((world,np.ones(5))).T
        images.append((p[:2]/p[2]).T)
    marker_iter=iter(images)
    monkeypatch.setattr(module,'marker_centres',lambda image:next(marker_iter))
    pose=reconstruct_pose(np.zeros((512,512,3),np.uint8),np.zeros((512,512,3),np.uint8))
    np.testing.assert_allclose(pose[:3],[5,0,.69],atol=1e-10)
    np.testing.assert_allclose(pose[7:],[0,0],atol=1e-10)


def test_onboard_horizon_and_gyro_propagation_are_image_only():
    h=w=128;yy,xx=np.indices((h,w));angle=.08
    image=np.zeros((h,w,3),np.uint8);image[yy>=64-np.tan(angle)*(xx-64)]=40
    senses=OwnSenses(3.,.01,0.,(.2,0.,0.))
    vision=OnboardVision();first=vision.features(image,senses,.02)
    assert first[0]==pytest.approx(angle,abs=.004)
    later=vision.features(None,senses,.02)
    assert later[0]==pytest.approx(first[0]+.004)


def test_runtime_policy_accepts_no_simulator_or_truth_argument():
    policy=ObservedRidingPolicy(np.ones((7,3)))
    rider=VideoRidingPolicy(policy)
    image=np.zeros((128,128,3),np.uint8);image[64:]=40
    action,features=rider.action(image,OwnSenses(3.,0.,0.,(0.,0.,0.)),.02)
    np.testing.assert_allclose(action,np.append(features,1.)@policy.weights)


def test_learning_ignores_poisoned_expert_audit_attributes(monkeypatch):
    import pai.bike.video_learning as module
    def motion(episode,*unused):
        x=np.random.default_rng(3).normal(size=(20,6));y=x[:,:3]/10
        return x,y,{'valid_marker_frames':20,'observed_window_mask':np.ones(20,bool)}
    monkeypatch.setattr(module,'decode_motion',motion)
    episode=PixelEpisode(np.zeros((20,2,16,16,3),np.uint8),np.arange(20)*.02)
    first,_=fit_video_policy([episode])
    object.__setattr__(episode,'audit_expert_states',np.full((20,11),999.))
    object.__setattr__(episode,'audit_expert_controls',np.full((20,3),-999.))
    second,_=fit_video_policy([episode])
    np.testing.assert_array_equal(first.weights,second.weights)
