"""Sensor-only arm policy interface; task scoring is deliberately separate."""
import numpy as np

from pai.config import load_config
from pai.envs import TabletopEnv
from pai.envs.disturbances import build_disturbances
from pai.envs.predicates import perceived_predicates
from pai.perception.camera_frame import CameraFramePerception
from pai.world.entities import POS, FORCE, GRIP, entity_names

CONDITIONS = ('none', 'push', 'heavier_object', 'slippery_object', 'camera_shift')


def observation(tokens, q, dq, previous_action, elapsed):
    """Given entity tokens + proprioception; fixed physical scaling, no labels."""
    x = tokens.copy()
    x[:, POS] /= .5
    x[0, FORCE] = np.arcsinh(x[0, FORCE]/10.)
    relative = np.r_[tokens[1, POS]-tokens[0, POS], tokens[5, POS]-tokens[1, POS]]/.3
    return np.clip(np.r_[x.ravel(), q/3., dq/3., previous_action, relative, elapsed/150.], -10, 10).astype(np.float32)


def potential(tokens):
    hand, obj, target = tokens[0, POS], tokens[1, POS], tokens[5, POS]
    distance = np.linalg.norm(hand-obj)
    reach = 1-np.tanh(10*distance)
    closed = np.clip((.065-float(tokens[0, GRIP][0]))/.04, 0, 1)
    grasp = closed*np.exp(-distance**2/.0025)
    lift = np.clip((obj[2]-.02)/.10, 0, 1)
    align = 1-np.tanh(10*np.linalg.norm(obj[:2]-target[:2]))
    return float(reach+2*grasp+lift*(2*grasp+3*align))


class ArmTask:
    """Identical physics/action rate/cap/settle and true success test as the slice.

    Only the policy input, dense reward, and perceived-goal early termination are
    constructed here. `score` is called after termination, never by the policy.
    """
    def __init__(self):
        self.cfg = load_config('configs/tabletop.yaml', ['env.render_images=false'])
        self.env = TabletopEnv(self.cfg.env, disturbances=[])
        self.perceive = CameraFramePerception(self.env)
        self.names = entity_names(self.env)
        self.steps = 0

    def reset(self, seed, disturbances):
        self.env.disturbances = build_disturbances(disturbances)
        self.obs = self.env.reset(seed=seed)
        self.perceive.reset()
        self.previous = np.zeros(4, np.float32)
        self.tokens = self.perceive(self.obs, self.obs['ee_pos'])
        self.steps = 0; self.phi = potential(self.tokens); self.returns = 0.
        return self.vector()

    def vector(self):
        return observation(self.tokens, self.obs['q'], self.obs['dq'], self.previous, self.steps)

    def step(self, action):
        action = np.clip(np.asarray(action), -1, 1)
        physical = action*np.array([.25,.25,.25,1.])
        for _ in range(self.cfg.slice.action_repeat):
            previous_ee = self.obs['ee_pos'].copy()
            self.obs = self.env.step(np.r_[physical[:3],0.,physical[3]])
        self.tokens = self.perceive(self.obs, previous_ee)
        self.steps += 1; self.previous = action.copy()
        perceived = perceived_predicates(self.env.objects, self.tokens, self.names)
        reached = ('on','red','plate') in perceived and float(self.tokens[0,GRIP][0]) > .065
        phi = potential(self.tokens)
        reward = .99*phi-self.phi-.01-.002*float(np.square(action[:3]).sum())+20.*reached
        self.phi = phi; self.returns += reward
        done = reached or self.steps >= self.cfg.slice.max_plan_steps
        truncated = done and not reached
        # Bootstrap at the last controlled state on timeout; settle is a passive
        # scoring tail, exactly the standard 25 control steps for all RL episodes.
        vector = self.vector()
        if done:
            for _ in range(self.cfg.slice.settle_steps):
                self.obs = self.env.step(np.array([0.,0.,0.,0.,1.]))
        return vector, float(reward), bool(done), bool(truncated)

    def score(self):
        return dict(success=('on','red','plate') in self.obs['predicates'], steps=self.steps,
            return_=float(self.returns), events=list(self.env.events.events))

    def close(self):
        self.env.close()
