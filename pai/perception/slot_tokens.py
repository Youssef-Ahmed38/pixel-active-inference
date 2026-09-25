"""From pixels to entity tokens (week 3): the drop-in replacement for privileged object state.

    camera frame -> frozen DINOv2 patch features -> object slots -> per-slot token readout
                 -> each known object is read from its own slot (slots trained with a fixed slot per
                    object), or matched to the slot whose predicted kind and colour fit best

Only the objects come from vision. The gripper token (position, opening, wrist force) comes from
proprioception and touch: a body senses itself directly. Object velocities are finite differences
of the perceived positions. With a fixed slot per object, identity is the slot index; with
interchangeable slots, each object is matched by its predicted kind and colour (and a checkpoint
trained for it can start each frame's slot attention from the previous frame's slots).

The output has exactly the layout of `pai.world.entities.encode`, so the world model, goals, planner
and cause inference run unchanged on perceived instead of privileged state. The ablation
"privileged vs perceived" is then a single switch.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from pai.perception.features import DinoFeatures
from pai.perception.slots import SlotModel
from pai.world.entities import COLOR, FORCE, GRIP, KIND, KINDS, POS, TOKEN_DIM, VEL, YAW, encode


class SlotPerception:
    def __init__(self, env, slot_ckpt: str, device: str = "cuda", camera: str | None = None):
        ckpt = torch.load(slot_ckpt, map_location=device, weights_only=False)
        self.model = SlotModel(**ckpt["model_config"]).to(device).eval()
        self.model.load_state_dict(ckpt["model"])
        self.tok_mean, self.tok_std = ckpt["tok_mean"].to(device), ckpt["tok_std"].to(device)
        self.dino = DinoFeatures(device=device)
        # the camera and resolution the slots were trained on (older checkpoints: front, 224)
        self.camera = camera or ckpt.get("camera", "front")
        self.image_size = int(ckpt.get("image_size", 224))
        # slots trained with a fixed slot per label: object o is always slot object_labels[0] + o
        self.fixed_start = int(ckpt.get("object_labels", [2])[0]) if ckpt.get("assignment") == "fixed" else None
        self.temporal = bool(ckpt.get("temporal_init", False))
        pca = ckpt.get("pca")
        self.pca = None if pca is None else (torch.as_tensor(pca["mean"], device=device),
                                             torch.as_tensor(pca["components"], device=device))
        self.env, self.device = env, device
        # What each known object should look like, from the scene description (kind + colour).
        ref = encode(env, env._observe())  # kind and colour of each object, from the scene description
        self.identity = torch.as_tensor(np.concatenate([ref[1:, KIND], ref[1:, COLOR]], -1), device=device)
        self.reset()

    # Finger opening when the hand has closed on something: an empty hand closes to ~0, a fully
    # open one is at 0.08 m. In between, the fingers are blocked by an object: touch, not vision.
    BLOCKED = (0.005, 0.065)
    HELD_RADIUS = 0.04  # an object seen this close to the hand when the fingers close is the one held
    OCCLUSION_RADIUS = 0.06  # the hand hides objects within this distance below it
    OCCLUSION_SHIFT = 0.03   # the largest apparent shift a partial view causes; beyond it, it moved
    SLIP_CLOSE = 0.015  # fingers closing this much further than at the grasp: the object is sliding out
    # Recalibrating the camera from pixels: off. Perception noise raised false 'camera moved' alarms in
    # 3-5 of 20 clean episodes (results/pixel_v2), and acting on them cost more than it gained: the
    # slot readout maps image to table directly, and camera-shift episodes succeeded 90% without it.
    allow_recalibration = False

    def reset(self) -> None:
        self.prev_slots = None
        self.prev_pos = None
        self.prev_t = None
        self.held = None  # (object index, offset from the hand) while an object is in the hand
        # the agent's own recalibration after it concluded "my camera moved" (added to everything
        # seen, as in pai.perception.camera_frame); the slice agent updates it
        self.correction = np.zeros(3, np.float32)

    def __call__(self, obs: dict, prev_ee: np.ndarray | None = None) -> np.ndarray:
        """The `perceive` interface of the slice agent: observation -> entity tokens."""
        image = self.env.render(self.camera)  # the perception camera, not necessarily the one in obs
        tokens, self.last_match_cost = self.perceive(image, obs, prev_ee)
        return tokens

    @torch.no_grad()
    def perceive(self, image: np.ndarray, obs: dict, prev_ee: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """image (S, S, 3) uint8 from `self.camera` at the training resolution; obs for proprioception.
        Returns tokens (N, TOKEN_DIM) and a match cost per object (lower = more confident)."""
        feats = self.dino(image[None]).reshape(1, -1, self.dino.model.embed_dim).float()
        if self.pca is not None:
            feats = (feats - self.pca[0]) @ self.pca[1]
        # Slots start from their learned queries unless the checkpoint was trained to start from the
        # previous frame's slots: untrained, they drift away within a few frames (held-out object
        # IoU 0.65 -> 0.04). With a fixed slot per object, identity needs no temporal link anyway.
        out = self.model(feats, init=self.prev_slots if self.temporal else None)
        self.prev_slots = out["slots"]
        slot_tok = out["tokens"][0] * self.tok_std + self.tok_mean              # (K, TOKEN_DIM), raw units
        looks = torch.cat([slot_tok[:, KIND], slot_tok[:, COLOR]], -1)          # (K, 7)
        cost = torch.cdist(self.identity, looks).cpu().numpy()                  # (n_objects, K)
        if self.fixed_start is not None:
            obj_idx = np.arange(len(self.identity))
            slot_idx = obj_idx + self.fixed_start
        else:
            obj_idx, slot_idx = linear_sum_assignment(cost)

        tokens = encode(self.env, obs, prev_ee)                                 # gripper from the body itself
        perceived = tokens.copy()
        pos = np.zeros((len(self.identity), 3), np.float32)
        for o, k in zip(obj_idx, slot_idx):
            t = slot_tok[k].cpu().numpy()
            pos[o] = t[POS] + self.correction
            perceived[1 + o, POS] = pos[o]
            perceived[1 + o, YAW] = t[YAW] / max(1e-6, float(np.linalg.norm(t[YAW])))
        # An object in the hand is sensed through the body: the fingers stopped closing on it, and
        # from then on it moves with the hand at the offset it had when grasped. The camera cannot
        # do this well (the hand hides it; lifted, its height is not in the image), and a wrong
        # "it fell" belief makes the agent open its hand and re-grasp forever.
        ee, grip = tokens[0, POS], float(obs["gripper"])
        blocked = self.BLOCKED[0] < grip < self.BLOCKED[1]
        if self.held is not None and (not blocked or grip < self.held[2] - self.SLIP_CLOSE):
            self.held = None       # opened, closed on nothing, or the fingers close further: it slides out
        if self.held is None and blocked:
            ref = self.prev_pos if self.prev_pos is not None else pos           # before this frame's occlusion
            d = np.linalg.norm(ref - ee, axis=1)
            o = int(np.argmin(d))
            if d[o] < self.HELD_RADIUS:
                self.held = (o, ref[o] - ee, grip)
        if self.held is not None:
            o, off, _ = self.held
            pos[o] = ee + off
            perceived[1 + o, POS] = pos[o]
        # Object permanence: an object under the hand is partly hidden by it, which shifts its seen
        # centre by 1-2 cm, exactly where the grasp needs it most. While the hand is over it, the last
        # clear estimate is kept, unless the camera sees it clearly elsewhere (more than the shift a
        # partial view causes): then the hand has pushed it, and the camera is right.
        if self.prev_pos is not None:
            for o in range(len(pos)):
                if self.held is not None and o == self.held[0]:
                    continue
                d_xy = np.linalg.norm(self.prev_pos[o, :2] - ee[:2])
                moved = np.linalg.norm(pos[o, :2] - self.prev_pos[o, :2]) > self.OCCLUSION_SHIFT
                if d_xy < self.OCCLUSION_RADIUS and ee[2] > self.prev_pos[o, 2] and not moved:
                    pos[o] = self.prev_pos[o]
                    perceived[1 + o, POS] = pos[o]
        # velocities over the time since the last perception (the agent perceives once per planner
        # step, i.e. every action_repeat control steps, not every control step)
        t = self.env.t
        if self.prev_pos is None or self.prev_t is None or t <= self.prev_t:
            perceived[1:, VEL] = 0.0
        else:
            perceived[1:, VEL] = (pos - self.prev_pos) / ((t - self.prev_t) * self.env.control_dt)
        self.prev_pos, self.prev_t = pos, t
        perceived[1:, GRIP] = 0.0
        perceived[1:, FORCE] = 0.0
        assert perceived.shape[-1] == TOKEN_DIM and KINDS[0] == "gripper"
        return perceived, cost[obj_idx, slot_idx]
