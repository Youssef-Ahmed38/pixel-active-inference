"""Generic motor skills for every embodiment (the given part of docs/DISCOVERY.md).

Skills act on a generic `Target` (a site or body frame with an axis and an approach direction), never on
a fixture type, through the common embodiment action; each returns a `SkillResult` with its own success
criterion and a `stalled` flag. What a skill does to the world is for the agent to learn.

    ctx = SkillContext(env, env.reset())
    h = handle_target(env.fixture)
    for name, kw in [("reach", {}), ("grasp", {}), ("push_pull", {"direction": "pull", "distance": 0.15})]:
        print(run_skill(ctx, name, h, **kw).row())
"""

from pai.skills.core import Recorder, SkillContext, SkillResult, StallMonitor, Target, handle_target
from pai.skills.primitives import grasp, insert, push_pull, reach, release, retreat, turn
from pai.skills.registry import SKILLS, Option, SkillDef, options, run_skill

__all__ = ["SKILLS", "Option", "Recorder", "SkillContext", "SkillDef", "SkillResult", "StallMonitor", "Target",
           "grasp", "handle_target", "insert", "options", "push_pull", "reach", "release", "retreat", "run_skill",
           "turn"]
