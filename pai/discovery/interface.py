"""The contract between the discovery agent (the mind) and a world it acts in (the body + scene).

The agent never sees what things *are* (a knob, a key, a keyhole). It sees:

    entities   what is in view: parts of fixtures (things attached to the world that may move
               about a joint) and free objects (things it can pick up), each with generic
               attributes only: which way it can move (hinge / slide / none), whether it is
               graspable, where it is
    features   a generic, discretised description of the observable state, e.g.
                   "part:p3.angle"  -> "rest" | "pos" | "neg"        (a hinge part's joint)
                   "part:p1.open"   -> "closed" | "ajar" | "open"    (a door or drawer)
                   "obj:o2.where"   -> "free" | "held" | "in:p5"     (a free object)
                   "obj:o2.angle"   -> "rest" | "pos" | "neg"        (an object turned in place)
               The *same* feature names mean the same kind of thing in every scene, but no
               feature says "locked" or "matching key": that is what the agent has to find out.
    actions    what it can try now: a skill (from pai.skills) applied to an entity, e.g.
               ("pull", "p1"), ("turn_pull", "p1", +1), ("pick", "o2"), ("insert", "o2", "p5"),
               ("turn_held", "o2", -1). Which actions exist follows from the entity attributes,
               never from what the entity is.
    outcome    after an action: did the skill itself run (reached, grasped), did it stall, which
               features changed, which new entities came into view (e.g. inside a drawer)

The goal is given as a feature condition, e.g. {"part:p1.open": "open"} for "open the door".

Two worlds implement this: a fast symbolic MockWorld (pai.discovery.mock) with the same
mechanics as the physics, for developing and testing the agent, and PhysicsWorld
(pai.discovery.physics) over the MuJoCo home-door scene and pai.skills.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

# Action kinds (the first element of an Action tuple). A world offers only those that apply.
PROBE_KINDS = ("pull", "push", "turn_pull", "turn_push", "press_push")  # move a part, optionally
#   turning (or pressing) it first and holding it there while moving it: the only way to move a
#   spring-latched door, which is exactly what the agent does not know
MANIPULATE_KINDS = ("turn", "pick", "insert", "turn_held", "release")

Action = tuple  # (kind, entity_id[, arg]) e.g. ("turn", "p3", -1), ("insert", "o2", "p5")


@dataclass
class Entity:
    id: str                        # opaque: "p1", "o2"; stable within an episode, meaningless across
    kind: str                      # "part" | "object"
    joint: str = "none"            # "hinge" | "slide" | "none": how it can move
    graspable: bool = True
    pos: tuple = (0.0, 0.0, 0.0)   # world position (m), for costs and for describing it
    attrs: dict = field(default_factory=dict)  # generic look: {"shape": "round", "size": 0.03, "colour": ...}


@dataclass
class Outcome:
    action: Action
    executed: bool                 # the skill ran to its end (reached, grasped where needed)
    stalled: bool                  # commanded motion but no progress (something holds it)
    changed: dict[str, tuple]      # feature -> (before, after)
    revealed: list[Entity]         # entities that came into view
    cost: float                    # time taken (s)
    notes: str = ""


class World(Protocol):
    def entities(self) -> list[Entity]: ...
    def features(self) -> dict[str, str]: ...
    def actions(self) -> list[Action]: ...
    def execute(self, action: Action) -> Outcome: ...
    def goal(self) -> dict[str, str]: ...
    def goal_reached(self) -> bool: ...
    def truth(self) -> dict: ...   # ground truth for scoring only: never read by the agent
