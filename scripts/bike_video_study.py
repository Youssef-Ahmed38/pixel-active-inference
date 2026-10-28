"""Pixel-only bicycle apprenticeship and on-board-sense riding, with truth audits."""
import argparse,base64,gzip,hashlib,inspect,io,json,os,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from pai.bike.env import BicycleEnv,BalanceController
from pai.bike.observation_learning import ObservedRidingPolicy
from pai.bike.video_sensors import VideoBicycleEnv,proprioception
from pai.bike.video_learning import PixelEpisode,decode_motion,fit_video_policy,fit_decoded_policy,VideoRidingPolicy
from pai.cognition.contracts import SkillCommand


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):
    p=Path(path);return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
def save(path,data):
    p=Path(path);blob=json.dumps(data,separators=(',',':'),allow_nan=False).encode()
    if re.search(rb'[A-Za-z]:[\\/]',blob) or b'/Users/' in blob or b'/home/' in blob:raise ValueError('absolute path in result')
    if len(blob)>5_000_000 and p.suffix!='.gz':raise ValueError('gzip required')
    if p.suffix=='.gz':blob=gzip.compress(blob,mtime=0)
    tmp=Path(str(p)+'.tmp');tmp.write_bytes(blob);os.replace(tmp,p)


def sources():
    return {p.as_posix():sha(p) for p in sorted([*Path('pai/bike').glob('*.py'),
      *(p for p in Path('scripts').glob('bike_video_*.py') if p.name!='bike_video_report.py')])}


def start(env,seed):
    rng=np.random.default_rng(seed);speed=float(rng.uniform(2.5,4.5));roll=float(rng.uniform(-.1,.1))
    env.reset(seed,speed,roll);return speed,roll


def pixel_file(seed):
    p=Path(f'results/bike_video_demo_{seed}.npz')
    return p if p.exists() else Path(str(p)+'.gz')


def pixels(seed):
    p=pixel_file(seed);raw=gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes()
    with np.load(io.BytesIO(raw)) as d:return PixelEpisode(d['rgb'],d['times'])


def collect(seeds):
    for seed in seeds:
        if pixel_file(seed).exists():print('Retaining teaching video',seed,flush=True);continue
        env=VideoBicycleEnv();speed,roll=start(env,seed);expert=BalanceController()
        frames=[];times=[];audit=[]
        with mujoco.Renderer(env.model,height=512,width=512) as renderer:
            while not env.done:
                views=[]
                for name in ('video0','video1'):
                    renderer.update_scene(env.data,camera=name);views.append(renderer.render().copy())
                frames.append(views);times.append(env.steps*env.control_dt)
                action=expert.action(env.sensation())
                # These labels live in a separate audit artifact, never in PixelEpisode.
                audit.append({'qpos':env.data.qpos.tolist(),'qvel':env.data.qvel.tolist(),
                              'sensation':env.sensation().tolist(),'action':action.tolist()})
                env.execute(SkillCommand('ride',parameters=tuple(map(float,action))))
        buf=io.BytesIO();np.savez_compressed(buf,rgb=np.asarray(frames),times=np.asarray(times))
        blob=buf.getvalue();p=Path(f'results/bike_video_demo_{seed}.npz')
        if len(blob)>5_000_000:p=Path(str(p)+'.gz');blob=gzip.compress(blob,mtime=0)
        p.write_bytes(blob)
        save(f'results/bike_video_audit_{seed}.json.gz',{'seed':seed,'samples':audit,
             'initial_speed':speed,'initial_roll':roll,'success':not env.fallen and env.travelled>=env.distance,
             'note':'Scoring only; fitting never opens this file','pixels_sha256':sha(p)})
        print('RGB teaching',seed,len(frames),'frames',p.stat().st_size,'bytes',flush=True)


def decoded(seed,inverse):
    # This cache is derived ONLY from the RGB file. The separate audit file is
    # never opened here. Fingerprints reject stale motion extraction.
    key=hashlib.sha256((sha(pixel_file(seed))+sha('pai/bike/video_sensors.py')+
                        inspect.getsource(decode_motion)+inverse).encode()).hexdigest()
    path=Path(f'results/bike_video_estimated_{inverse}_{seed}.npz')
    if path.exists():
        with np.load(path) as d:
            if str(d['key'])==key:
                return d['features'].copy(),d['actions'].copy(),{'valid_marker_frames':int(d['valid']),
                  'observed_window_mask':d['mask'].copy(),'estimated_qpos':d['qpos'].copy(),'estimated_qvel':d['qvel'].copy()}
    print('Decoding RGB episode',seed,flush=True)
    x,y,info=decode_motion(pixels(seed),inverse)
    np.savez_compressed(path,key=key,features=x,actions=y,valid=info['valid_marker_frames'],
                       mask=info['observed_window_mask'],qpos=info['estimated_qpos'],qvel=info['estimated_qvel'])
    return x,y,info


def train(seeds,output,inverse='mujoco',feature_set='all'):
    if Path(output).exists():raise FileExistsError(output)
    policy,report=fit_decoded_policy((decoded(seed,inverse) for seed in seeds),inverse=inverse,feature_set=feature_set)
    policy.save(output)
    report.update({'seeds':seeds,'policy':output,'policy_sha256':sha(output),
                   'pixel_inputs':{pixel_file(s).as_posix():sha(pixel_file(s)) for s in seeds},
                   'sources':sources(),'expert_states_used_for_fitting':False,'expert_controls_used_for_fitting':False})
    save('results/bike_video_'+inverse+'_'+feature_set+'_training.json.gz',report)
    print('Fitted video policy',report['decision_examples'],'examples',policy.weights.tolist(),flush=True)


def own_trial(policy,seed,practice=False,video=None,record_senses=False):
    env=VideoBicycleEnv(distance=1e6 if practice else 10.,max_seconds=5. if practice else 10.)
    speed,roll=start(env,seed);rider=VideoRidingPolicy(policy);frames=[];trace=[];odo=0.;maximum_roll=0.
    initial_hash=hashlib.sha256(env.data.qpos.tobytes()+env.data.qvel.tobytes()).hexdigest()
    with mujoco.Renderer(env.model,height=128,width=128) as renderer:
        for step in range(250 if practice else 500):
            if not practice and env.done:break
            image=None
            if step%5==0 or video is not None:
                renderer.update_scene(env.data,camera='onboard');image=renderer.render().copy()
            policy_image=image if step%5==0 else None
            senses=proprioception(env);action,features=rider.action(policy_image,senses,env.control_dt)
            maximum_roll=max(maximum_roll,abs(float(features[0])))
            if maximum_roll>.6 and practice:break
            odo+=max(0.,senses.wheel_speed*np.cos(features[5]))*env.control_dt
            row={'t':step*env.control_dt,'features':features.tolist(),'action':action.tolist()}
            if record_senses:
                row['proprio']={'wheel_speed':senses.wheel_speed,'steering':senses.steering,
                               'torso_lean':senses.torso_lean,'gyro':list(senses.gyro)}
                row['rgb_png']=None
                if policy_image is not None:
                    import imageio.v2 as imageio
                    row['rgb_png']=base64.b64encode(imageio.imwrite('<bytes>',policy_image,format='png')).decode()
            trace.append(row)
            if video is not None:frames.append(image)
            env.execute(SkillCommand('ride',parameters=tuple(map(float,action))))
            if practice and odo>=10:break
    # Practice optimizer sees only this self-sensed tuple, never the audit below.
    sensed_score=(int(odo>=10 and maximum_roll<=.6),min(odo,10.)-.2*maximum_roll)
    audit={'seed':seed,'start_speed':speed,'start_roll':roll,'success':bool(not env.fallen and env.travelled>=10),
           'initial_state_sha256':initial_hash,
           'distance':env.travelled,'fallen':env.fallen,'seconds':float(env.data.time),
           'sensed_odometry':odo,'sensed_maximum_roll':maximum_roll,'own_trace':trace}
    if video is not None:
        import imageio.v2 as imageio
        imageio.mimsave(video,frames,fps=50,codec='libx264',quality=6,macro_block_size=1)
        if Path(video).stat().st_size>=3_000_000:raise ValueError('video must stay below 3 MB')
    return sensed_score,audit


def evaluate_dev(policy_path,seeds,output):
    policy=ObservedRidingPolicy.load(policy_path);rows=[]
    for seed in seeds:
        _,r=own_trial(policy,seed);rows.append(r)
        print('Own-camera development',seed,r['success'],round(r['distance'],2),flush=True)
    save(output,{'policy':policy_path,'policy_sha256':sha(policy_path),'episodes':rows,'sources':sources()})


def practice(policy_path,seeds,updates,output):
    if Path(output).exists():raise FileExistsError(output)
    best=ObservedRidingPolicy.load(policy_path);rng=np.random.default_rng(15000);history=[]
    def score(policy):
        pairs=[own_trial(policy,s,practice=True) for s in seeds]
        return (sum(p[0][0] for p in pairs),sum(p[0][1] for p in pairs)/len(pairs)),[p[1] for p in pairs]
    best_score,rows=score(best);history.append({'attempt':0,'score':best_score,'accepted':True,'audits':rows})
    scales=np.array([.5,.12,.03])
    for attempt in range(1,updates+1):
        candidate=ObservedRidingPolicy(best.weights+rng.normal(size=best.weights.shape)*scales,
                     'video initialization; practice reward from own odometry and camera/gyro roll only')
        candidate.weights[[3,4]]=0.  # Preserve the given no-joint-feedback architecture.
        candidate_score,rows=score(candidate);accepted=candidate_score>best_score
        if accepted:best,best_score=candidate,candidate_score
        history.append({'attempt':attempt,'score':candidate_score,'accepted':accepted,'audits':rows})
        save('results/bike_video_practice.json.gz',{'optimizer_seed':15000,'practice_seeds':seeds,'history':history,
             'reward_inputs':'own wheel odometry, camera/gyro roll; ground-truth audit fields excluded from selection'})
        print('Own-sense practice',attempt,accepted,best_score,flush=True)
    best.save(output)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('collect','train','dev','practice'))
    p.add_argument('--seeds',nargs='+',type=int,default=list(range(12000,12004)))
    p.add_argument('--policy',default='results/bike_video_imitation.json');p.add_argument('--updates',type=int,default=40)
    p.add_argument('--inverse',choices=('mujoco','servo'),default='mujoco')
    p.add_argument('--features',choices=('all','no_joints','no_joints_constant_lean'),default='all')
    p.add_argument('--output',default='results/bike_video_development.json.gz');a=p.parse_args()
    if a.mode=='collect':collect(a.seeds)
    elif a.mode=='train':train(a.seeds,a.policy,a.inverse,a.features)
    elif a.mode=='dev':evaluate_dev(a.policy,a.seeds,a.output)
    else:practice(a.policy,a.seeds,a.updates,a.output)


if __name__=='__main__':main()
