"""How each body grasps each class of home-door part: found by search on development seeds.

    GRASPS[body][part class] = {"roll": rad, "flip": 0 | 1, "spin": rad, "off": (dx, dy, dz), "grip": -1..1}

Part classes: "knob_round", "knob_ball", "lever" (the door handles), "thumb" (the thumb-turn wing),
"drawer" (the drawer bar), "key" (a key bow on a cradle: table or shelf), "key_drawer" (a key bow in the
open drawer). The palm orientation is HomeDoorOracle._grasp_R(part bar axis, approach, roll, flip, spin):
the approach (towards the part: into the door, or down onto a key) rolled by `roll` about the bar, palm y
along +/- the bar, and for a knob the bar turned by `spin` across its face. `off` is the grasp centre's
offset from the part in the palm frame; `grip` the closing command. A class missing for a body is grasped
the way that fits best kinematically (pai.skills.clearance); the table holds the grasps that search
(scripts/home_door_demo.py --calibrate) found to work better than that on the development seeds.
"""

GRASPS: dict[str, dict[str, dict]] = {}
