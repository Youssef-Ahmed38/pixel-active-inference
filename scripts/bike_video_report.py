"""Append the completed pixel-only teaching / on-board riding study."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.bike_video_study import load,save,sha,decoded
from scripts.bike_video_final import PLAN,DATA


def teaching_audit():
    rows=[]
    for seed in range(12000,12004):
        x,y,info=decoded(seed,'servo');truth=load(f'results/bike_video_audit_{seed}.json.gz')
        mask=info['observed_window_mask']
        sensations=np.array([r['sensation'] for r in truth['samples']])
        actions=np.array([r['action'] for r in truth['samples']]);q=np.array([r['qpos'] for r in truth['samples']])
        difference=x-sensations;difference[:,[0,3,4,5]]=(difference[:,[0,3,4,5]]+np.pi)%(2*np.pi)-np.pi
        rows.append({'seed':seed,'frames':len(x),'supported_windows':int(mask.sum()),'expert_goal_success':truth['success'],
         'feature_mae':np.abs(difference[mask]).mean(0).tolist(),'inferred_action_mae':np.abs(y[mask]-actions[mask]).mean(0).tolist(),
         'root_position_mae_m':np.abs(info['estimated_qpos'][mask,:3]-q[mask,:3]).mean(0).tolist()})
    result={'episodes':rows,'note':'Post-fit scoring only; expert states and controls never entered fitting or model selection',
            'feature_order':['roll','roll_rate','speed','steering','torso_lean','yaw'],
            'action_order':['torque','steer_target','lean_target']}
    path='results/bike_video_teaching_audit.json.gz'
    if Path(path).exists():assert load(path)==result,'stored teaching audit differs'
    else:save(path,result)
    print('Teaching audit reproduced; no policy fitting',flush=True)


def main():
    plan=load(PLAN);data=load(DATA);summary=load('results/bike_video_final_summary.json.gz')
    audit=load('results/bike_video_final_audit.json.gz');teaching=load('results/bike_video_teaching_audit.json.gz')
    media=load('results/bike_video_media.json.gz');practice=load('results/bike_video_practice.json.gz')
    assert data['complete'] and audit['passed']
    accepted=[r['attempt'] for r in practice['history'] if r['accepted'] and r['attempt']>0]
    failures={c:[{'seed':r['seed'],'distance':r['distance'],'reason':'fall' if r['fallen'] else 'timeout'}
                for r in data['episodes'] if r['condition']==c and not r['success']] for c in plan['conditions']}
    # Seed-paired descriptive differences; no replacement of the registered outcomes.
    rng=np.random.default_rng(17000);comparisons={}
    for other in ('expert','state_student','hybrid_student','video_practiced'):
        indexed={(r['seed'],r['condition']):r for r in data['episodes']}
        differences=np.array([int(indexed[(s,'video_imitation')]['success'])-int(indexed[(s,other)]['success']) for s in plan['seeds']])
        samples=rng.integers(0,20,(5000,20));gains=differences[samples].mean(1)
        comparisons[other]={'mean_gain':float(differences.mean()),'paired_start_bootstrap95':np.quantile(gains,[.025,.975]).tolist(),
                            'imitation_only_success':int((differences==1).sum()),'other_only_success':int((differences==-1).sum())}
    save('results/bike_video_report.json.gz',{'summary':summary['summary'],'failures':failures,'paired':comparisons,
        'accepted_practice_attempts':accepted,'plan_sha256':sha(PLAN),'data_sha256':sha(DATA),
        'media':media,'audit':audit,'scope':'calibrated marker-based stereo teaching; real attached on-board camera and proprioception'})
    lines=['\n## Video-only apprenticeship and attached-camera riding (2026-10-28)\n',
      '**Frozen final seeds 16000–16019:** the video imitation student reaches the 10-metre goal on '
      '**20/20** starts from its attached camera and proprioception. The practiced video branch reaches '
      '**19/20**, so practice did not improve held-out success. The existing state student is 20/20, '
      'hybrid 18/20, and classical expert 12/20 on the same physical starts. This is a limited, strongly '
      'calibrated video benchmark, not unstructured real-world video learning or a complete Stage 7 pass.\n',
      '### What is learned and what is given\n',
      '`video_sensors.py` adds five **massless coloured visual markers**, two fixed external cameras '
      '(512×512, 50 Hz), and a camera rigidly mounted on the frame. The cameras/markers do not change '
      'body masses, inertias, actuators or contacts. Fixed-camera pixel centroids are triangulated, then '
      'rigid alignment and known hinge axes reconstruct motion. No expert pose/control label is an input '
      'to this reconstruction or policy fit. Learning files contain only `rgb` and `times`; the expert '
      'states and controls are in separate audit files opened only for post-fit scoring. Cached states '
      'are **RGB estimates**, fingerprinted against their pixel source and decoder, not simulator states.\n',
      'Given: tag identities/geometry, calibrated stereo and on-board cameras, known initial joint rest, '
      'wheel radius and rolling approximation, MuJoCo dynamics, servo gains/damping/inertia, actuator limits, '
      'a geometric horizon/road decoder, and the linear no-joint-feedback policy architecture. The pose '
      'decoder is authored inverse graphics, **not a learned neural pose encoder**. Learned: policy weights '
      'from visually inferred targets, plus optional practice mutations selected from own-sensor fitness. '
      'There is no expert-action cloning, supervised expert-state fitting, locomotion transfer, or cloud training.\n',
      'Four expert recordings (12000–12003) contain 958 synchronized time frames / 1,916 RGB views. '
      'One expert recording times out and remains included. An 11-frame cubic smoother uses future frames '
      'only during offline demonstration processing. Missing markers cause abstention/interpolation; the fit '
      'uses only 540 fully observed temporal windows and weights each teaching ride equally. Robust ridge '
      'regression uses inferred actions only. The two joint-feedback rows are fixed to zero as a given '
      'architecture: including current joint angles encouraged trivial copying of the reconstructed servo '
      'targets instead of learning balance. All remaining coefficients, including their signs, are fitted.\n',
      'The inverse remains **given and approximate**. The selected low-bandwidth inverse uses rolling '
      'longitudinal inertia and each servo’s own inertia, gravity and damping; it does not identify full '
      'contact dynamics from video. Reconstructed wheel rotation assumes rolling without slip. The original '
      'full constrained inverse was too sensitive to video-rate acceleration and contact-height noise.\n',
      '### Development failures retained\n',
      '| Candidate | Development seeds 13000–13002 | Result |','|---|---|---|',
      '| Full inverse, unrestricted feedback | 0/3 goals | Failed at about 1.3–1.5 m |',
      '| Servo inverse, unrestricted feedback | 0/3 goals | About 2.8–5.0 m |',
      '| Servo inverse, no joint feedback | 3/3 goals | Selected before final evaluation |',
      '| No joint feedback, constant torso target | 0/3 goals | Rejected |',
      '\nThe selected imitation branch also passes 13003–13005. No final layout was used for this selection. '
      'See the `bike_video_development_*.json.gz` artifacts; failed variants were not overwritten.\n',
      '### Own senses at deployment\n',
      'The frame-mounted camera is addressed by its fixed camera ID, not by a root-tracking `lookat`. '
      '128×128 RGB arrives at 10 Hz; wheel/joint encoders and the body gyro arrive at 50 Hz. Wheel speed '
      'is encoder rotation × 0.32 m, **not root linear velocity**. Absolute lean comes from the visible '
      'ground/sky horizon, heading from the marked lane’s vanishing point, and gyro integration bridges '
      'camera samples. The policy receives only RGB plus an `OwnSenses` record; it cannot access the env, '
      'root pose/velocity, simulator quaternion, world coordinates, or a tracking camera. Joint encoders '
      'are available but the selected architecture does not use their feedback in the high-level action.\n',
      'This benchmark gives a black sky, flat ground, white lane edges and ideal simulated proprioception. '
      'The learner has not been shown to work on arbitrary scenery, camera calibration, lighting, noise, '
      'unmarked monocular teaching video, wheel slip, starts from rest, obstacles or a full humanoid rider.\n',
      '### Practice uses the same senses\n',
      f"Optimizer seed 15000, starts 14000–14005, 24 mutations plus the initial policy: 150 practice rides. "
      f"Accepted mutation attempts: `{accepted}`. The six sensed goals were already reached at attempt zero. "
      'Fitness uses integrated wheel speed × cos(image/gyro heading), maximum camera/gyro lean, and a '
      'given 0.6-rad own-lean cutoff. Simulator displacement, fall and success fields are audit-only. '
      'This self-odometry objective is approximate and is not a claim of sample-efficiency improvement. '
      'The mutation that improved practice fitness introduced a held-out failure at 16006.\n',
      '### Frozen paired evaluation\n',
      'Each condition uses the same twenty initial states: speed uniform 2.5–4.5 m/s, roll ±0.1 rad, '
      '10 m forward displacement, ten-second limit. Initial state hashes match across all five riders. '
      'The legacy expert/state/hybrid run in their original scene; the new scene adds only nonphysical '
      'visual aids. The legacy hybrid retains its root-tracking camera and externally supplied speed, '
      'so it is explicitly privileged. The state/hybrid policies are the existing optimizer-1000 '
      'practiced student and its matched 16×16 RGB model, not newly fitted baselines.\n',
      '| Rider | Goals reached | Success / Wilson 95% interval |','|---|---:|---|']
    for c,r in summary['summary'].items():
        lo,hi=r['wilson95'];lines.append(f"| {c} | {r['successes']}/{r['episodes']} | {100*r['success_rate']:.1f}% [{100*lo:.1f}, {100*hi:.1f}] |")
    lines+=['\nThese are twenty shared starts, not 100 independent layouts. A 20/20 sample is not proof of '
            '100% reliability. Both video branches were registered before final collection; neither was '
            'retuned or substituted after seeing final failures.\n',
            '| Failed rider | Seeds and failure type |','|---|---|']
    for c,rs in failures.items():
        if rs:lines.append(f"| {c} | "+', '.join(f"{r['seed']} ({r['reason']}, {r['distance']:.2f} m)" for r in rs)+' |')
    lines+=['\n### Post-fit teaching audit (not fitting labels)\n',
            '| Teaching seed | Supported windows | Roll MAE (rad) | Speed MAE (m/s) | Inferred torque / steer / torso target MAE |',
            '|---|---:|---:|---:|---|']
    for r in teaching['episodes']:
        a=r['inferred_action_mae'];lines.append(f"| {r['seed']} | {r['supported_windows']} | {r['feature_mae'][0]:.4f} | "
             f"{r['feature_mae'][2]:.4f} | {a[0]:.3f} Nm / {a[1]:.4f} rad / {a[2]:.4f} rad |")
    lines+=['\nThe inferred actions are imperfect, especially torso targets; successful riding must not be '
            'described as exact expert-control recovery. Audit fields were first compared after the '
            'policies and final run were frozen.\n',
            '### Truth boundary, video, and limitations\n',
            f"All {audit['sensor_only_episodes_replayed']} new-rider episodes replay from their actual "
            f"{audit['raw_onboard_frames_replayed']} lossless RGB frames and proprioception, with identical "
            'features/actions to 1e-10 tolerance and no scoring-state input. Every practice score and '
            'accept/reject decision is recomputed from its own-sensed fields. Tests also forbid root '
            'array access in the encoder adapter, verify physical camera mounting and unchanged dynamics, '
            'and alter expert audit attributes without changing fitted weights. All 26 bike tests pass.\n',
            f"On-board clip: [video-only student ride]({Path('media/bike/video_onboard_student.mp4').as_posix()}), "
            f"seed {media['seed']}, {media['frames']} original sensor frames at {media['fps']} Hz, "
            f"{media['bytes']} bytes. It is the first successful practiced final episode (otherwise the "
            'first failure); it was encoded from recorded decision images without rerunning the environment. '
            'The failed practiced episode remains in the results.\n',
            'This closes the calibrated marker-based video apprenticeship / attached-camera moving-start '
            'test. It does not establish natural-video motion discovery, learned inverse dynamics, an '
            'unstructured learned vision encoder, broad cycling competence, Stage 2 transfer, or reliable '
            'benefit from practice. Existing checkpoints and default bike behavior remain unchanged.\n',
            '### Seeds and reproduction\n',
            'Teaching 12000–12003; development/model selection 13000–13005; practice starts 14000–14005; '
            'practice optimizer 15000; frozen final starts 16000–16019. Static camera debug seed 20000 '
            'was development only. The final plan inventories recorded previous bike seed fields and '
            'rejects overlap. Models, pixel inputs, cache lineage and final source hashes are retained.\n',
            '```powershell',"$env:PYTHONPATH='.'","$env:OMP_NUM_THREADS='1'","$env:OPENBLAS_NUM_THREADS='1'",
            '.venv/Scripts/python.exe scripts/bike_video_study.py collect --seeds 12000 12001 12002 12003',
            '.venv/Scripts/python.exe scripts/bike_video_study.py train --seeds 12000 12001 12002 12003 --inverse servo --features no_joints --policy results/bike_video_no_joints.json',
            '.venv/Scripts/python.exe scripts/bike_video_study.py dev --policy results/bike_video_no_joints.json --seeds 13000 13001 13002 --output results/bike_video_development_no_joints.json.gz',
            '.venv/Scripts/python.exe scripts/bike_video_study.py dev --policy results/bike_video_no_joints.json --seeds 13003 13004 13005 --output results/bike_video_development_additional.json.gz',
            '.venv/Scripts/python.exe scripts/bike_video_study.py practice --policy results/bike_video_no_joints.json --seeds 14000 14001 14002 14003 14004 14005 --updates 24 --output results/bike_video_practiced.json',
            '.venv/Scripts/python.exe scripts/bike_video_final.py register',
            '.venv/Scripts/python.exe scripts/bike_video_final.py run',
            '.venv/Scripts/python.exe scripts/bike_video_final.py score',
            '.venv/Scripts/python.exe scripts/bike_video_final.py audit',
            '.venv/Scripts/python.exe scripts/bike_video_final.py media',
            '.venv/Scripts/python.exe scripts/bike_video_report.py teaching-audit',
            '.venv/Scripts/python.exe scripts/bike_video_report.py',
            '.venv/Scripts/python.exe -m pytest tests/test_bike*.py -q --basetemp=data/pytest_bike_video_new',
            '```\n',
            'Use clean output paths for reproduction. Collection/final resumption retain completed '
            'episodes and reject changed frozen parameters; teaching fits do not overwrite existing '
            'models. On this managed Windows host, the actual pytest run wrapped a fresh directory '
            'with inherited permissions because mkdir(mode=0700) is inaccessible; the recorded test '
            'results are `bike_video_tests_frozen.json.gz`. No existing temp directories were removed.\n',
            'Artifacts under `results/bike_video_*`: RGB `.npz` (gzip-wrapped above 5 MB), separate '
            'truth audits, visually estimated motion caches, training/development failures, both '
            'policy checkpoints, every practice attempt, frozen plan, 100 final episode records '
            'including raw camera/proprioception traces, summary, replay audit and media provenance. '
            'Result paths are project-relative; no user folders or cache paths are recorded.']
    doc=Path('docs/BIKE_STAGE7.md');old=doc.read_text(encoding='utf-8')
    if '## Video-only apprenticeship and attached-camera riding (2026-10-28)' in old:raise ValueError('section already exists')
    doc.write_text(old+'\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',nargs='?',choices=('report','teaching-audit'),default='report')
    args=parser.parse_args();teaching_audit() if args.mode=='teaching-audit' else main()
