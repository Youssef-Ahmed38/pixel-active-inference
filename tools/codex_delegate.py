"""Delegate a build task to OpenAI Codex (ChatGPT's coding agent) and let Claude supervise it.

    python tools/codex_delegate.py start  --task maze-v3 --brief docs/briefs/MAZE_V3.md
    python tools/codex_delegate.py status --task maze-v3
    python tools/codex_delegate.py resume --task maze-v3 --feedback review.md
    python tools/codex_delegate.py finish --task maze-v3      # remove the worktree after merging

How it works
------------
* Each task gets its own git worktree and branch (``codex/<task>``) next to the repository, cut
  from ``main``. Codex works only there, so it can never switch the main checkout's branch or mix
  its edits with other work in progress.
* Codex runs non-interactively (``codex exec``) in the background with a workspace-write sandbox
  rooted at the worktree. Its prompt is the brief plus fixed rules (stay in your files, no git
  commands, honest numbers, a structured final report).
* Everything is logged under ``<worktree>/.codex_task/``: the prompt, the JSON event log, the final
  report (``report.md``) and the session id used by ``resume``.
* The supervisor (Claude) reviews the worktree (diff, tests, recomputed numbers, leakage), then
  either sends feedback with ``resume`` or commits, pushes the branch and opens a pull request.

The Codex executable is found automatically (newest ``codex.exe`` under ``~/.codex``) or given
with ``--codex`` / the ``CODEX_EXE`` environment variable.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RULES = """
## Rules for this delegated task (from the supervisor)

- You work in a git worktree created for this task. Edit only the files the brief allows.
- Do NOT run git commands that change state (commit, push, checkout, switch, branch, merge, rebase,
  reset, stash, worktree). The supervisor commits after reviewing your work.
- Use the project's Python: `{python}`. Run from the worktree root with `PYTHONPATH=.` so the
  worktree's code is imported (check once: `python -c "import pai; print(pai.__file__)"`).
- Training runs on this laptop's GPU (8 GB) and CPU only; never use Kaggle or cloud compute.
- Honest evaluation: final numbers on fresh seeds never used during development, frozen
  parameters, exact commands; keep failures and negative results; no ground truth in the agent.
- Commits are not yours to make, so do not add any author or co-author lines anywhere.
- Finish with a final message in this format (it is saved as the task report):
  ## Summary
  ## Files changed
  ## Commands run (tests and evaluations)
  ## Measured results (with seeds)
  ## What is not done / known problems
"""


def find_codex(explicit: str | None) -> Path:
    if explicit or os.environ.get("CODEX_EXE"):
        return Path(explicit or os.environ["CODEX_EXE"])
    found = sorted((Path.home() / ".codex").rglob("codex.exe"), key=lambda p: p.stat().st_mtime)
    if not found:
        sys.exit("codex.exe not found under ~/.codex; pass --codex or set CODEX_EXE")
    return found[-1]  # the most recently installed copy (older copies may not support the model)


def worktree(task: str) -> Path:
    return REPO.parent / f"pai-codex-{task}"


def git(*args: str, cwd: Path = REPO) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def link(src: Path, dst: Path) -> None:
    """Junction (Windows) or symlink to large ignored folders the task may need to read."""
    if dst.exists() or not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(src)], check=True, capture_output=True)
    else:
        dst.symlink_to(src, target_is_directory=True)


THREAD_RULES = """
- You are continuing your existing conversation about this project, working directly in the main
  checkout `{repo}` on branch `main` (not in a separate worktree, whatever the rules above say).
  Stay on `main`: never switch branches, stash, reset or commit; the supervisor works in the same
  folder and commits only reviewed paths.
- Touch only the files your brief allows; other uncommitted files in the folder belong to other work.
"""


def start_in_thread(args) -> None:
    """Continue the user's existing Codex thread in the main checkout: it keeps its context, and the
    user watches it live in the Codex app (codex exec resume writes into the same conversation).
    Codex cannot combine resume with a worktree, so the brief's rules keep it on main."""
    meta = REPO / "data" / "codex_tasks" / args.task
    meta.mkdir(parents=True, exist_ok=True)
    brief = Path(args.brief).read_text(encoding="utf-8")
    python = REPO / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    prompt = brief + RULES.format(python=python) + THREAD_RULES.format(repo=REPO)
    (meta / "prompt.md").write_text(prompt, encoding="utf-8")
    (meta / "thread").write_text(args.thread)
    if git("branch", "--show-current").strip() != "main":
        sys.exit("the main checkout is not on main; refusing to start")
    codex = find_codex(args.codex)
    cmd = [str(codex), "exec", "resume", args.thread, "--json", "-m", args.model, "--enable", "code_mode_host",
           "-c", 'sandbox_mode="workspace-write"', "-o", str(meta / "report.md"), "-"]
    with open(meta / "prompt.md", "rb") as stdin, open(meta / "events.jsonl", "ab") as out:
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS if os.name == "nt" else 0
        proc = subprocess.Popen(cmd, cwd=REPO, stdin=stdin, stdout=out, stderr=subprocess.STDOUT,
                                creationflags=flags, start_new_session=os.name != "nt")
    (meta / "pid").write_text(str(proc.pid))
    (meta / "worktree").write_text(str(REPO))
    print(f"Codex continues thread {args.thread} (pid {proc.pid}) in {REPO}; watch it in the Codex app\n"
          f"log: {meta / 'events.jsonl'}", flush=True)
    wait(proc, meta, args.wait)


def start(args) -> None:
    if args.thread:
        return start_in_thread(args)
    wt = worktree(args.task)
    if not wt.exists():
        git("worktree", "add", "-b", f"codex/{args.task}", str(wt), args.base)
    for name in args.link:  # read access to big ignored folders, e.g. third_party, runs/slots
        link(REPO / name, wt / name)
    meta = wt / ".codex_task"
    meta.mkdir(exist_ok=True)
    brief = Path(args.brief).read_text(encoding="utf-8")
    python = REPO / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    prompt = brief + RULES.format(python=python)
    (meta / "prompt.md").write_text(prompt, encoding="utf-8")
    codex = find_codex(args.codex)
    cmd = [str(codex), "exec", "-C", str(wt), "-s", "workspace-write", "--json", "-m", args.model, "--enable", "code_mode_host",
           "-o", str(meta / "report.md"), "--add-dir", str(REPO / ".venv"), "-"]
    with open(meta / "prompt.md", "rb") as stdin, open(meta / "events.jsonl", "ab") as out:
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS if os.name == "nt" else 0
        proc = subprocess.Popen(cmd, stdin=stdin, stdout=out, stderr=subprocess.STDOUT, creationflags=flags,
                                start_new_session=os.name != "nt")
    (meta / "pid").write_text(str(proc.pid))
    print(f"Codex started (pid {proc.pid}) in {wt} on branch codex/{args.task}\nlog: {meta / 'events.jsonl'}",
          flush=True)
    wait(proc, meta, args.wait)


def wait(proc, meta: Path, enabled: bool) -> None:
    """--wait: stay attached until Codex ends. Without it, a caller whose shell ends (e.g. a
    supervisor's tool call) takes the child process with it; run the whole command as a tracked
    background job with --wait instead."""
    if not enabled:
        return
    code = proc.wait()
    report = meta / "report.md"
    ready = report.exists() and report.stat().st_size > 0
    print(f"Codex exited with code {code}; report {'ready' if ready else 'MISSING'} ({report})", flush=True)


def session_id(meta: Path) -> str | None:
    for line in (meta / "events.jsonl").read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        sid = ev.get("thread_id") or ev.get("session_id") or (ev.get("msg") or {}).get("session_id")
        if sid:
            return sid
    return None


def running(meta: Path) -> bool:
    pid = (meta / "pid").read_text().strip() if (meta / "pid").exists() else ""
    if not pid:
        return False
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
        return pid in out
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def locate(task: str) -> tuple[Path, Path]:
    """(worktree, meta folder) of a task: a thread task keeps its logs in data/codex_tasks/<task>."""
    meta = REPO / "data" / "codex_tasks" / task
    if meta.exists():
        wt = Path((meta / "worktree").read_text().strip()) if (meta / "worktree").exists() else REPO
        return wt, meta
    wt = worktree(task)
    return wt, wt / ".codex_task"


def status(args) -> None:
    wt, meta = locate(args.task)
    if not meta.exists():
        sys.exit(f"no task {args.task}")
    print(f"worktree: {wt}\nrunning: {running(meta)}\nsession: {session_id(meta)}")
    print(f"report: {'ready' if (meta / 'report.md').exists() else 'not yet'} ({meta / 'report.md'})")
    print(git("status", "--short", cwd=wt) or "(no changes yet)")


def resume(args) -> None:
    wt, meta = locate(args.task)
    sid = (meta / "thread").read_text().strip() if (meta / "thread").exists() else session_id(meta)
    if not sid:
        sys.exit("no session id in the log; cannot resume")
    if running(meta):
        sys.exit("Codex is still running on this task")
    feedback = Path(args.feedback).read_text(encoding="utf-8")
    (meta / "report.md").unlink(missing_ok=True)
    codex = find_codex(args.codex)
    cmd = [str(codex), "exec", "resume", sid, "--json", "-m", args.model, "--enable", "code_mode_host", "-o", str(meta / "report.md"), "-"]
    with open(args.feedback, "rb") as stdin, open(meta / "events.jsonl", "ab") as out:
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS if os.name == "nt" else 0
        proc = subprocess.Popen(cmd, cwd=wt, stdin=stdin, stdout=out, stderr=subprocess.STDOUT,
                                creationflags=flags, start_new_session=os.name != "nt")
    (meta / "pid").write_text(str(proc.pid))
    print(f"Codex resumed session {sid} (pid {proc.pid}) with {len(feedback)} characters of feedback", flush=True)
    wait(proc, meta, args.wait)


def finish(args) -> None:
    wt = worktree(args.task)
    shutil.rmtree(wt / ".codex_task", ignore_errors=True)
    git("worktree", "remove", "--force", str(wt))
    print(f"removed {wt} (branch codex/{args.task} kept)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["start", "status", "resume", "finish"])
    ap.add_argument("--task", required=True, help="short name, e.g. maze-v3")
    ap.add_argument("--brief", help="start: the task brief (markdown)")
    ap.add_argument("--feedback", help="resume: the supervisor's review (markdown)")
    ap.add_argument("--base", default="main")
    ap.add_argument("--link", nargs="*", default=["third_party"], help="ignored folders to link read-only")
    ap.add_argument("--codex", help="path to codex.exe")
    ap.add_argument("--wait", action="store_true", help="stay attached until Codex ends (see wait())")
    ap.add_argument("--model", default="gpt-6.1-sol", help="Codex model (default gpt-6.1-sol, the model the "
                                                          "project thread was recorded with)")
    ap.add_argument("--thread", help="start: continue this existing Codex thread in the main checkout "
                                     "(keeps its context; visible live in the Codex app)")
    args = ap.parse_args()
    {"start": start, "status": status, "resume": resume, "finish": finish}[args.action](args)


if __name__ == "__main__":
    main()
