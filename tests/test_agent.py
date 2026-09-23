import numpy as np
import pytest
import torch

from pai.agents import PixelAIAgent
from pai.config import Config, load_config, REPO_ROOT
from pai.envs import PandaEnv


def agent_cfg(**kw):
    base = load_config(REPO_ROOT / "configs" / "default.yaml").agent.to_dict()
    base.update(kw)
    return Config(base)


class LinearDecoder(torch.nn.Module):
    """g(mu) = sigmoid(W mu + b) reshaped to an image: a smooth, known generative model."""

    def __init__(self, n: int, size: int = 16, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.image_size = size
        self.W = torch.nn.Parameter(torch.randn(3 * size * size, n, generator=g))
        self.b = torch.nn.Parameter(torch.zeros(3 * size * size))

    def forward(self, mu):
        return torch.sigmoid(mu @ self.W.T + self.b).view(-1, 3, self.image_size, self.image_size)


def _img(dec, q):
    with torch.no_grad():
        x = dec(torch.as_tensor(q, dtype=torch.float32)[None])[0]
    return (x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def test_visual_perception_corrects_wrong_belief():
    """With proprioception off, vision alone must pull a wrong belief towards the truth."""
    dec = LinearDecoder(4)
    q_true = np.array([0.3, -0.2, 0.1, 0.4])
    cfg = agent_cfg(goal_mode="joints", pi_q=0.0, pi_v=200.0, beta=0.0, k_a=0.0)
    agent = PixelAIAgent(cfg, dec)
    agent.reset(q_true, goal_q=q_true, mu0=q_true + 0.3)
    obs = {"q": q_true, "image": _img(dec, q_true)}
    err0 = np.linalg.norm(agent.mu.numpy() - q_true)
    for _ in range(400):
        agent.step(obs)
    assert np.linalg.norm(agent.mu.numpy() - q_true) < 0.2 * err0


def test_image_goal_attractor_points_towards_goal():
    dec = LinearDecoder(4)
    q_goal = np.array([0.5, 0.0, -0.3, 0.2])
    mu = np.zeros(4)
    agent = PixelAIAgent(agent_cfg(goal_mode="image", pi_v=0.0), dec)
    agent.reset(mu, goal_image=_img(dec, q_goal))
    _, f, *_ = agent._visual_terms(None)
    assert float(f @ torch.as_tensor(q_goal - mu, dtype=torch.float32)) > 0


def test_action_follows_belief():
    """If the belief is ahead of the measured joint angles, the action pushes towards it."""
    cfg = agent_cfg(goal_mode="joints", pi_v=0.0, beta=0.0)
    agent = PixelAIAgent(cfg)
    q = np.zeros(4)
    agent.reset(q, goal_q=q, mu0=np.array([0.2, -0.2, 0.0, 0.1]))
    a, _ = agent.step({"q": q, "image": None})
    assert a[0] > 0 and a[1] < 0 and a[3] > 0


@pytest.mark.slow
def test_joint_goal_reaching_in_sim():
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml",
                      ["env.image_size=64", "agent.goal_mode=joints", "agent.pi_v=0"])
    env = PandaEnv(cfg.env, disturbances=[])
    rng = np.random.default_rng(3)
    q0, qg = env.sample_q(rng), env.sample_q(rng)
    obs = env.reset(q0=q0)
    agent = PixelAIAgent(cfg.agent)
    agent.reset(obs["q"], goal_q=qg)
    for _ in range(250):
        a, _ = agent.step(obs)
        obs = env.step(a)
    assert np.linalg.norm(obs["ee_pos"] - env.ee_pos_at(qg)) < 0.01
    env.close()
