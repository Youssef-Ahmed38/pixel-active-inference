"""Fetch robot models from MuJoCo Menagerie (sparse checkout, pinned commit).

The Franka Panda drives the tabletop scene; the hands and the humanoid are the embodiments of the
door/drawer scene (pai/envs/embodiments.py). Idempotent: an existing checkout only gets the missing
model directories added to its sparse set.
"""

import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/google-deepmind/mujoco_menagerie.git"
COMMIT = "367e3d9884401dcf6f9c27fa69f118992539039f"
DEST = Path(__file__).resolve().parent.parent / "third_party" / "mujoco_menagerie"
MODELS = ["franka_emika_panda", "robotiq_2f85", "unitree_g1", "wonik_allegro", "leap_hand", "shadow_hand"]


def run(*cmd, cwd=None):
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)


def main():
    missing = [m for m in MODELS if not (DEST / m).is_dir()]
    if not missing:
        print(f"already present: {DEST}")
        return
    if not (DEST / ".git").exists():
        DEST.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--filter=blob:none", "--sparse", "--no-checkout", REPO, str(DEST))
        run("git", "sparse-checkout", "set", *MODELS, cwd=DEST)
        run("git", "checkout", COMMIT, cwd=DEST)
        return
    # Existing checkout: extend the sparse set (keeps the pinned commit), then re-apply the checkout.
    run("git", "sparse-checkout", "set", *MODELS, cwd=DEST)
    run("git", "checkout", COMMIT, cwd=DEST)


if __name__ == "__main__":
    sys.exit(main())
