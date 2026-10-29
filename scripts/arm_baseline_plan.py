"""Immutable registration and provenance for mainstream arm baselines."""
import hashlib
import importlib.metadata
import json
import re
import time
from pathlib import Path

from pai.config import load_config

PREFIX='results/arm_baselines_'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    text=json.dumps(value,indent=2,allow_nan=False)
    if re.search(r'(?<![A-Za-z])[A-Za-z]:[\\/]|/Users/|/home/',text):raise ValueError('Absolute local path in result')
    path=Path(path);tmp=Path(str(path)+'.tmp');tmp.write_text(text,encoding='utf8')
    for i in range(100):
        try:tmp.replace(path);return
        except PermissionError:
            if i==99:raise
            time.sleep(.05)


def sources():
    return {p.as_posix():digest(p) for pattern in ('pai/baselines/*.py','scripts/arm_baseline_*.py') for p in Path('.').glob(pattern)}


def register():
    path=Path(PREFIX+'plan.json')
    if path.exists():return json.loads(path.read_text(encoding='utf8'))
    cfg=load_config('configs/tabletop.yaml')
    p=dict(registered_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        methods=['ppo','recurrent_ppo','full_agent','no_adapt'],training_seeds=[120000,120001,120002],
        final_seed_base=110000,final_n=100,development_seed_base=111000,development_n=50,
        calibration_seeds=list(range(112000,112008)),disturbance_rng_bases=dict(final=115000,development=116000),
        train_layout_rule='10000000 + seed_index*1000000 + worker*100000 + episode_counter; conditions uniform independently',
        budget_steps=1000448,n_envs=4,rollout=256,epochs=6,minibatch=256,sequence=32,
        lr=.0003,gamma=.99,gae_lambda=.95,clip=.2,entropy=.01,value_coefficient=.5,grad_clip=.5,target_kl=.03,
        plateau='After750592 steps, stop if best of last4 logged rolling100 returns fails to exceed best of preceding4 by1% of max(1,abs(previousbest)); evaluated every25600 steps',
        selection='Last completed update under fixed budget/plateau; no final or development checkpoint selection',
        observations='158 floats: default privileged camera-frame entity tokens, q/dq, previous action, own relative hand/object/plate features, elapsed fraction; no cause labels, event logs, disturbance parameters or scripted phase',
        reward='Given potential shaping .99*Phi(next)-Phi(current) -.01 -.002*motion effort +20 for perceived on(red,plate) with open hand. Phi=reach+2*softgrasp+lift*(2*softgrasp+3*alignment). All from own tokens; true task success only scores.',
        timing='Actions every5 control steps (.1s); xyz velocities±.25m/s, grip±1, yaw0; cap150; perceived goal+open hand early termination;25 passive open/still control steps before true success test',
        final_policy='Deterministic tanh(mean); recurrent hidden reset per episode; all models frozen before final physical collection',
        statistics='Per-condition/overall Wilson95 per run; mean±sampleSD over3 neural seeds. Exact two-sided paired sign test on discordant successes vs full per seed; no pooled pseudo-independent count. Also no-adapt vs full.',
        optional_model_based_rl=False,config=cfg.to_dict(),world_model_sha256=digest(cfg.slice.world_model),
        agent_sources={p:digest(p) for p in ('pai/agents/slice_agent.py','pai/envs/tabletop.py','pai/perception/camera_frame.py','pai/causes/inference.py','configs/tabletop.yaml')},
        versions={p:importlib.metadata.version(p) for p in ('torch','numpy','mujoco','scipy')})
    save(path,p)
    Path('docs/ARM_BASELINES.md').write_text('''# Arm learning baselines under hidden disturbances

## Registered protocol (before training)

Compare full agent, no-adaptation agent, clipped PPO MLP and recurrent PPO GRU for
on(red,plate), in the five standard conditions. Privileged object tokens are given,
through the same default camera-frame observation used by the published arm, alongside
q/dq and wrist force. Policy inputs exclude event logs, disturbance labels/parameters,
true predicates, and the arm's authored subgoal phase. Derived relative positions and
fixed physical input scaling are given. PPO outputs continuous Cartesian velocity and
gripper commands; no scripted controller, demonstrations or action residual are supplied.

Dense potential shaping is given, based only on perceived reach, soft grasp, lift and
alignment; it is not learned. The policy learns its entire action mapping. Termination
uses perceived placement with an open hand, or the same 150-decision cap as the slice;
25 passive settling controls precede the unchanged true task success test. The arm's
own subgoal termination differs in detail; all use the same physics, action rate/limits,
maximum time and success predicate. Realized disturbance exposure is reported, including
scheduled disturbances that occur after an early finish.

Each learned baseline has three optimizer seeds120000–120002, at most1,000,448
decision transitions per seed (4 environments,256-step rollouts). Standard PPO settings:
Adam3e-4 linearly annealed,6 epochs,minibatch256,clip.2,gamma.99,GAE.95,
entropy.01,value coefficient.5,gradient clip.5,target approximateKL.03. Recurrent
BPTT uses32-step chunks with stored initial hidden states and explicit episode resets.
Training stops at budget or the registered rolling-return plateau after750592 steps.
No final-based tuning or checkpoint selection; the last completed update is frozen.
Detailed settings and exact stopping definition are in results/arm_baselines_plan.json.

Training layouts use disjoint ranges beginning10000000; the environment independently
draws each condition uniformly and uses the unchanged slice disturbance distribution.
Development layouts111000–111049 are reserved for diagnostics, not final scoring.
Final layouts110000–110099 are100 fresh episodes,20 per condition, identical for
every method and optimizer seed. Disturbance RNG base115000 plus episode index.
Independent nominal calibration112000–112007 is used only by the Bayesian agent.
Final policies are deterministic and all weights/settings freeze before final collection.

Report per-condition and overall success with Wilson95 intervals, learned mean±SD
over seeds, and exact paired sign tests against full on discordant episode outcomes,
separately per training seed. No pooling three predictions of the same episode as
independent observations. Full/no-adapt also have cause accuracy and their own English
episode explanations; PPO policies have no cause/explanation head. Keep all failures.

This is a comparison of whole systems, not an isolated test of cause representations:
the full agent has given relational subgoals, a learned model trained partly from
scripted demonstrations, authored likelihoods and adaptation rules. PPO receives a
given shaped reward and domain-randomized practice without demonstrations. A PPO
failure without adequate clean-task learning is an optimization/budget limitation,
not evidence that explicit causes are intrinsically superior. Count the world model's
training data and optimizer steps in sample efficiency, with provenance limitations.
Optional Dreamer/TD-MPC is not registered for this run. Laptop only; no git state changes.
''',encoding='utf8')
    return p


if __name__=='__main__':print(json.dumps(register(),indent=2))
