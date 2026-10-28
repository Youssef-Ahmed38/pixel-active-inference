"""Frozen paired starts, perceptual/teaching audits and actual on-board footage."""
import argparse,base64,hashlib,io,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import mujoco
from pai.bike.env import BicycleEnv,BalanceController
from pai.bike.observation_learning import ObservedRidingPolicy
from pai.bike.vision import ImageOrientation
from pai.bike.vision_control import ImageBalanceController
from pai.bike.video_sensors import OwnSenses
from pai.bike.video_learning import VideoRidingPolicy
from scripts.bike_video_study import load,save,sha,sources,start,own_trial,decoded

PLAN=Path('results/bike_video_final_plan.json.gz');DATA=Path('results/bike_video_final_episodes.json.gz')
STATE='runs/bike/practice_seed1000/practiced_observed_motion.json'
HYBRID='runs/bike/vision_practiced16/orientation.json'


def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0,center-radius),min(1,center+radius)]


def register():
    if PLAN.exists():raise FileExistsError(PLAN)
    paths=[STATE,HYBRID,'results/bike_video_no_joints.json','results/bike_video_practiced.json']
    used=set()
    def walk(x,key=''):
        if isinstance(x,dict):
            for k,v in x.items():walk(v,k)
        elif isinstance(x,list):
            for v in x:walk(v,key)
        elif isinstance(x,int) and not isinstance(x,bool) and 'seed' in key:used.add(x)
    for p in Path('results').glob('bike*.json*'):
        try:walk(load(p))
        except (ValueError,OSError):pass
    seeds=list(range(16000,16020));assert not set(seeds)&used
    save(PLAN,{'seeds':seeds,'sources':sources(),'models':{p:sha(p) for p in paths},
               'conditions':['expert','state_student','hybrid_student','video_imitation','video_practiced'],
               'initialization':'same seed, uniform speed 2.5–4.5 m/s, roll -0.1–0.1 rad',
               'sensor_schedule':'video: 128x128 attached-frame camera at 10 Hz; gyro and joint/wheel encoders at 50 Hz',
               'baseline_scene':'original BicycleEnv for original models; identical inertias/dynamics and starts',
               'given':'tag geometry, stereo/onboard calibration, horizon/lane decoder, rolling/servo inverse approximation, no-joint-feedback policy architecture',
               'rule':'Frozen parameters; run once; truth only scores; no promotion by final-seed tuning',
               'previously_recorded_seed_values':sorted(used)})
    print('Registered 20 fresh paired starts, five riders; no final episodes run yet',flush=True)


def legacy_trial(condition,seed):
    env=BicycleEnv();speed,roll=start(env,seed)
    initial_hash=hashlib.sha256(env.data.qpos.tobytes()+env.data.qvel.tobytes()).hexdigest()
    policy=BalanceController() if condition=='expert' else ObservedRidingPolicy.load(STATE)
    visual=ImageBalanceController(ImageOrientation.load(HYBRID),policy) if condition=='hybrid_student' else None
    renderer=None
    try:
        if visual:renderer=mujoco.Renderer(env.model,height=96,width=96)
        while not env.done:
            if visual:
                camera=mujoco.MjvCamera();camera.distance=3;camera.azimuth=120;camera.elevation=-15
                camera.lookat[:]=env.data.qpos[:3]  # Explicitly privileged OLD hybrid baseline only.
                renderer.update_scene(env.data,camera=camera)
                obs=env.sensation();action=visual.action(renderer.render(),env.data.time,obs[2],obs[3],obs[4])
            else:action=policy.action(env.sensation())
            from pai.cognition.contracts import SkillCommand
            env.execute(SkillCommand('ride',parameters=tuple(map(float,action))))
    finally:
        if renderer:renderer.close()
    return {'seed':seed,'start_speed':speed,'start_roll':roll,'success':bool(not env.fallen and env.travelled>=10),
            'initial_state_sha256':initial_hash,
            'distance':env.travelled,'fallen':env.fallen,'seconds':float(env.data.time)}


def run():
    plan=load(PLAN)
    assert plan['sources']==sources() and all(sha(p)==v for p,v in plan['models'].items())
    data=load(DATA) if DATA.exists() else {'plan_sha256':sha(PLAN),'episodes':[],'complete':False}
    assert data['plan_sha256']==sha(PLAN)
    done={(r['seed'],r['condition']) for r in data['episodes']}
    for seed in plan['seeds']:
        for condition in plan['conditions']:
            if (seed,condition) in done:continue
            if condition.startswith('video_'):
                path='results/bike_video_no_joints.json' if condition=='video_imitation' else 'results/bike_video_practiced.json'
                _,row=own_trial(ObservedRidingPolicy.load(path),seed,record_senses=True)
            else:row=legacy_trial(condition,seed)
            row['condition']=condition;data['episodes'].append(row);save(DATA,data)
            print(seed,condition,row['success'],round(row['distance'],2),flush=True)
    data['complete']=True;data['sources_unchanged']=sources()==plan['sources'];save(DATA,data)


def score():
    plan=load(PLAN);data=load(DATA);assert data['complete'] and data['sources_unchanged']
    summaries={}
    for c in plan['conditions']:
        rows=[r for r in data['episodes'] if r['condition']==c];n=len(rows);k=sum(r['success'] for r in rows)
        assert n==20 and len({r['seed'] for r in rows})==20
        summaries[c]={'successes':k,'episodes':n,'success_rate':k/n,'wilson95':wilson(k,n),
                      'mean_distance':float(np.mean([r['distance'] for r in rows]))}
    save('results/bike_video_final_summary.json.gz',{'plan_sha256':sha(PLAN),'data_sha256':sha(DATA),
        'summary':summaries,'note':'Wilson intervals describe 20 shared starts; conditions are paired, not independent samples'})
    print(summaries,flush=True)


def audit():
    import imageio.v2 as imageio
    plan=load(PLAN);data=load(DATA);assert data['complete'] and plan['sources']==sources()
    assert all(sha(p)==v for p,v in plan['models'].items())
    checks=0;frames=0
    for seed in plan['seeds']:
        hashes={r['initial_state_sha256'] for r in data['episodes'] if r['seed']==seed}
        assert len(hashes)==1
    for row in data['episodes']:
        if not row['condition'].startswith('video_'):continue
        path='results/bike_video_no_joints.json' if row['condition']=='video_imitation' else 'results/bike_video_practiced.json'
        actor=VideoRidingPolicy(ObservedRidingPolicy.load(path))
        for step in row['own_trace']:
            rgb=imageio.imread(io.BytesIO(base64.b64decode(step['rgb_png']))) if step['rgb_png'] else None
            senses=OwnSenses(**step['proprio']);action,features=actor.action(rgb,senses,.02)
            np.testing.assert_allclose(action,step['action'],rtol=0,atol=1e-10)
            np.testing.assert_allclose(features,step['features'],rtol=0,atol=1e-10)
            frames+=rgb is not None
        checks+=1
    practice=load('results/bike_video_practice.json.gz');best=None
    for attempt in practice['history']:
        audits=attempt['audits'];s=(sum(r['sensed_odometry']>=10 and r['sensed_maximum_roll']<=.6 for r in audits),
            sum(min(r['sensed_odometry'],10.)-.2*r['sensed_maximum_roll'] for r in audits)/len(audits))
        np.testing.assert_allclose(s,attempt['score'],atol=1e-10)
        accepted=best is None or s>best
        assert accepted==attempt['accepted']
        if accepted:best=s
    save('results/bike_video_final_audit.json.gz',{'passed':True,'sensor_only_episodes_replayed':checks,
          'raw_onboard_frames_replayed':frames,'practice_own_fitness_recomputed':True,
          'practice_attempts':len(practice['history']),'data_sha256':sha(DATA),'frozen_models_unchanged':True,
          'inference_inputs':'RGB PNGs + wheel/joint encoders + gyro; scoring labels excluded from replay'})
    print('Replayed',checks,'own-sense episodes,',frames,'raw frames; practice selection uses own fitness',flush=True)


def media():
    import imageio.v2 as imageio
    rows=[r for r in load(DATA)['episodes'] if r['condition']=='video_practiced']
    chosen=next((r for r in rows if r['success']),rows[0])
    frames=[imageio.imread(io.BytesIO(base64.b64decode(s['rgb_png']))) for s in chosen['own_trace'] if s['rgb_png']]
    path=Path('docs/media/bike/video_onboard_student.mp4');path.parent.mkdir(parents=True,exist_ok=True)
    imageio.mimsave(path,frames,fps=10,codec='libx264',quality=7,macro_block_size=1)
    assert path.stat().st_size<3_000_000
    save('results/bike_video_media.json.gz',{'path':path.as_posix(),'sha256':sha(path),'seed':chosen['seed'],
       'success':chosen['success'],'frames':len(frames),'fps':10,'bytes':path.stat().st_size,
       'selection':'first successful practiced episode, otherwise first failure; original recorded sensor frames, no rerun'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('register','run','score','audit','media'));a=p.parse_args()
    globals()[a.mode]()
