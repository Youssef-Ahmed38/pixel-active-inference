"""Fetch the Franka Panda model from MuJoCo Menagerie (sparse checkout, pinned commit)."""

import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/google-deepmind/mujoco_menagerie.git"
COMMIT = "367e3d9884401dcf6f9c27fa69f118992539039f"
DEST = Path(__file__).resolve().parent.parent / "third_party" / "mujoco_menagerie"


def run(*cmd, cwd=None):
    print("+", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)


def main():
    if (DEST / "franka_emika_panda" / "panda.xml").exists():
        print(f"already present: {DEST}")
        return
    DEST.parent.mkdir(parents=True, exist_ok=True)
    run("git", "clone", "--filter=blob:none", "--sparse", "--no-checkout", REPO, str(DEST))
    run("git", "sparse-checkout", "set", "franka_emika_panda", cwd=DEST)
    run("git", "checkout", COMMIT, cwd=DEST)


if __name__ == "__main__":
    sys.exit(main())
