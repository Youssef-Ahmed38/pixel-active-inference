import json

from pai.config import load_config, REPO_ROOT
from pai.eval.reach import evaluate


def test_joint_goal_eval_writes_summary(tmp_path):
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml", [
        "env.image_size=64", "agent.goal_mode=joints", "agent.pi_v=0",
        "eval.episodes=2", "eval.steps=150", "eval.save_gif=true", f"eval.out_dir={tmp_path}",
        "env.disturbances=[{type: push, start: 100, duration: 10, force: [0, 200, 0]}]",
    ])
    summary = evaluate(cfg, tag="t")
    assert summary["success_rate"] == 1.0
    assert summary["max_dev_after_success_median"] > cfg.eval.success_ee_dist  # the push is felt
    saved = json.loads((tmp_path / "t_reach.json").read_text())
    assert len(saved["episodes"]) == 2 and (tmp_path / "t_reach.gif").exists()
