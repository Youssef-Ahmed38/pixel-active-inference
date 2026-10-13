"""Code and run identities recorded by chess training/evaluation tools."""
import hashlib
import subprocess
import sys
from pathlib import Path

def metadata(seeds=(), time_controls=None):
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    digest = hashlib.sha256()
    for folder, pattern in ((root / "pai/chess", "*.py"), (root / "scripts", "chess_*.py")):
        for path in sorted(folder.glob(pattern)):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return {"git_commit": commit, "source_sha256": digest.hexdigest(), "seeds": list(seeds),
            "time_controls": time_controls or {}, "command": subprocess.list2cmdline(sys.argv)}
