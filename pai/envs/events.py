"""Ground-truth event log: the answer key for the agent's explanations.

Every control step it records, with timestamps:
- contact_start / contact_end between tracked bodies (objects, fingers, table)
- grasp / release (an object touching both fingers)
- slip (a grasped object moving relative to the hand)
- relation_true / relation_false (predicate changes, e.g. ("on", "red", "plate"))
- disturbance_start / disturbance_end (push, occlusion, mass change, ...)

Events are plain dicts, so episodes can be saved as JSON and compared with the agent's beliefs.
"""

from __future__ import annotations

import mujoco
import numpy as np

SLIP_SPEED = 0.10  # m/s of a grasped object relative to the grip site, sustained for DEBOUNCE steps
DEBOUNCE = 3       # control steps a grasp or relation change must persist before it is logged


class EventLog:
    def __init__(self, env):
        self.env = env
        m = env.model
        self.finger_ids = {m.body("left_finger").id, m.body("right_finger").id}
        self.table_geom = m.geom("table").id
        self.tracked = {env.obj_body[n]: n for n in env.objects}
        self.tracked.update({m.body("left_finger").id: "left_finger", m.body("right_finger").id: "right_finger",
                             m.body("hand").id: "hand"})
        self.reset()

    def reset(self) -> None:
        self.events: list[dict] = []
        self._contacts: set[frozenset] = set()
        self._grasped: set[str] = set()
        self._preds: set[tuple] = set()
        self._pending: dict[tuple, int] = {}  # candidate change -> consecutive steps seen
        self._dist_active = [False] * len(self.env.disturbances)
        self._contacts = self._current_contacts()
        self._grasped = self._compute_grasped(self._contacts)

    def grasped_objects(self) -> set[str]:
        return set(self._grasped)

    def _name(self, geom_id: int) -> str | None:
        if geom_id == self.table_geom:
            return "table"
        return self.tracked.get(int(self.env.model.geom_bodyid[geom_id]))

    def _current_contacts(self) -> set[frozenset]:
        d = self.env.data
        pairs = set()
        for i in range(d.ncon):
            c = d.contact[i]
            a, b = self._name(c.geom1), self._name(c.geom2)
            if a and b and a != b:
                pairs.add(frozenset((a, b)))
        return pairs

    def _compute_grasped(self, contacts: set[frozenset]) -> set[str]:
        out = set()
        for name in self.env.objects:
            if frozenset((name, "left_finger")) in contacts and frozenset((name, "right_finger")) in contacts:
                out.add(name)
        return out

    def _log(self, kind: str, **details) -> None:
        self.events.append({"t": self.env.t, "type": kind, **details})

    def update(self, obs: dict) -> None:
        env = self.env
        contacts = self._current_contacts()
        for pair in contacts - self._contacts:
            self._log("contact_start", bodies=sorted(pair))
        for pair in self._contacts - contacts:
            self._log("contact_end", bodies=sorted(pair))
        self._contacts = contacts

        raw = self._compute_grasped(contacts)
        for n in self._debounced("grasp", raw - self._grasped):
            self._log("grasp", object=n)
            self._grasped.add(n)
        for n in self._debounced("release", self._grasped - raw):
            self._log("release", object=n)
            self._grasped.discard(n)
        grasped = self._grasped

        grip_vel = np.zeros(6)
        mujoco.mj_objectVelocity(env.model, env.data, mujoco.mjtObj.mjOBJ_SITE, env.grip_site, grip_vel, 0)
        slipping = {n for n in grasped if np.linalg.norm(obs["state"][n]["vel"] - grip_vel[3:]) > SLIP_SPEED}
        for n in self._debounced("slip", slipping):  # brief relative motion while the fingers close is not a slip
            self._log("slip", object=n, speed=float(np.linalg.norm(obs["state"][n]["vel"] - grip_vel[3:])))

        # Spatial relations (near, left_of) change constantly; only task relations are logged.
        preds = {p for p in obs["predicates"] if p[0] not in ("near", "left_of")}
        for p in sorted(self._debounced("rel_true", preds - self._preds)):
            self._log("relation_true", relation=list(p))
            self._preds.add(p)
        for p in sorted(self._debounced("rel_false", self._preds - preds)):
            self._log("relation_false", relation=list(p))
            self._preds.discard(p)

        for i, d in enumerate(env.disturbances):
            active = d.active(env.t)
            if active != self._dist_active[i]:
                self._log("disturbance_start" if active else "disturbance_end",
                          disturbance=getattr(d, "kind", type(d).__name__), params=_jsonable(d.params))
                self._dist_active[i] = active

    def _debounced(self, kind: str, candidates: set) -> list:
        """Candidates that have persisted for DEBOUNCE consecutive steps; resets the others."""
        confirmed = []
        for c in candidates:
            key = (kind, c)
            self._pending[key] = self._pending.get(key, 0) + 1
            if self._pending[key] >= DEBOUNCE:
                confirmed.append(c)
                del self._pending[key]
        for key in [k for k in self._pending if k[0] == kind and k[1] not in candidates]:
            del self._pending[key]
        return confirmed

    def of_type(self, *kinds: str) -> list[dict]:
        return [e for e in self.events if e["type"] in kinds]


def _jsonable(params: dict) -> dict:
    return {k: (v.tolist() if isinstance(v, np.ndarray) else list(v) if isinstance(v, tuple) else v)
            for k, v in params.items()}
