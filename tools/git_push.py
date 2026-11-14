#!/usr/bin/env python3
"""Commit this copy of the project and push it to GitHub and GitLab in one go (the launcher's Maintenance ->
GitHub / GitLab runs this; it also works from a WSL terminal).

    python3 tools/git_push.py --check   --github URL --gitlab URL
    python3 tools/git_push.py --message "what changed" --github URL --gitlab URL [--branch main]

Steps: create the repository if there is none (`--init`), point the remotes "github" and "gitlab" at the URLs,
stage everything that .gitignore lets through, refuse if a confidential or oversized file is in it, commit, then
push to each remote. Both pushes are tried even if one fails; the exit code is 1 if any step failed.

Sign-in: Git for Windows is used when it is installed (from WSL through `git.exe`), so the Git Credential Manager
asks once in the browser for GitHub and for GitLab and remembers it. This script never sees or stores a password
or token. Without Git for Windows, WSL's git needs SSH keys (git@github.com:... URLs).

Kept out on purpose (team decision 13 Nov 2026, code + trained models): the CAD (cad/), the LR26 body and wheel
USDs (assets/lr26/*.usda) and the Performance department's LapSim (lapsim/LR26_LapSim/). They are listed in
.gitignore; CONFIDENTIAL below blocks a push even if someone edits .gitignore.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REMOTES = ("github", "gitlab")
CONFIDENTIAL = ("cad/*", "*.stp", "*.step", "assets/lr26/*.usda", "lapsim/LR26_LapSim/*")
MAX_FILE_MB = 95                 # GitHub refuses files over 100 MB
URL_RE = re.compile(r"^(https://[\w.-]+(:\d+)?/[\w./~-]+|[\w.-]+@[\w.-]+:[\w./~-]+|ssh://[\w.@:/~-]+|file://\S+|/\S+)$")
BRANCH_RE = re.compile(r"^[A-Za-z0-9][\w./-]{0,99}$")
HINTS = (
    ("non-fast-forward", "the remote already has commits this copy does not (e.g. a README created on the website): "
                         "create the repository empty, or merge it once by hand; this tool never force-pushes"),
    ("fetch first", "the remote already has commits this copy does not (e.g. a README created on the website): "
                    "create the repository empty, or merge it once by hand; this tool never force-pushes"),
    ("could not read username", "no sign-in available: install Git for Windows (Git Credential Manager), "
                                "or use an SSH URL with a key set up in WSL"),
    ("authentication failed", "sign-in refused: sign in again when the browser window opens, check the account "
                              "has write access, or remove the old entry in Windows Credential Manager"),
    ("repository not found", "check the URL and that the repository exists and you have write access"),
    ("permission denied (publickey)", "SSH key not accepted: add the WSL key to your account or use the https URL"),
    ("protected branch", "the branch is protected on the server: push to another branch or allow pushes"),
)


def is_confidential(path: str) -> bool:
    p = path.replace("\\", "/").lower()
    return any(fnmatch.fnmatch(p, pat.lower()) for pat in CONFIDENTIAL)


def hint_for(output: str) -> str:
    low = output.lower()
    for key, hint in HINTS:
        if key in low:
            return hint
    return ""


class Git:
    """Git for Windows (git.exe, with the Credential Manager) when WSL can run it, else the git on PATH."""

    def __init__(self, repo: Path, exe: str | None = None):
        self.repo = repo
        self.exe = exe or self.find()
        self.windows = self.exe.lower().endswith(".exe")
        self.where = self._win_path(repo) if self.windows else str(repo)
        if not self.where:
            raise RuntimeError(f"cannot translate {repo} into a Windows path for git.exe")

    @staticmethod
    def find() -> str:
        if os.environ.get("FSG_GIT"):
            return os.environ["FSG_GIT"]
        for c in ("/mnt/c/Program Files/Git/cmd/git.exe", "/mnt/c/Program Files/Git/bin/git.exe"):
            if os.path.isfile(c):
                return c
        return shutil.which("git.exe") or shutil.which("git") or ""

    @staticmethod
    def _win_path(p: Path) -> str:
        try:
            return subprocess.run(["wslpath", "-w", str(p)], capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    def run(self, *args: str, check: bool = False, quiet: bool = False) -> subprocess.CompletedProcess:
        # files exactly as on disk (no CRLF conversion; .gitattributes still applies), no terminal prompts (the
        # launcher has no terminal); the Credential Manager's browser sign-in still works
        cmd = [self.exe, "-C", self.where, "-c", "core.autocrlf=false", "-c", "core.safecrlf=false",
               "-c", "core.quotepath=false", *args]
        env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
        if self.windows:
            env["WSLENV"] = ":".join(x for x in (env.get("WSLENV", ""), "GIT_TERMINAL_PROMPT") if x)
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
                           stdin=subprocess.DEVNULL)
        if not quiet:
            out = (r.stdout + r.stderr).strip()
            if out:
                print("\n".join("    " + line for line in out.splitlines()), flush=True)
        if check and r.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed ({r.returncode})")
        return r


def problems_in_index(git: Git, repo: Path, untracked: bool = False) -> list[str]:
    """Confidential or too large files that the next commit would contain (untracked: also files not added yet
    that .gitignore lets through, for --check)."""
    files = git.run("ls-files", "-z", *(["-c", "-o", "--exclude-standard"] if untracked else []), quiet=True).stdout.split("\0")
    bad = []
    for f in filter(None, files):
        if is_confidential(f):
            bad.append(f"{f}: confidential (kept out of git on purpose)")
            continue
        try:
            mb = (repo / f).stat().st_size / 1e6
        except OSError:
            continue
        if mb > MAX_FILE_MB:
            bad.append(f"{f}: {mb:.0f} MB (GitHub refuses files over 100 MB)")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--github", default="", help="GitHub repository URL (https://github.com/<team>/<repo>.git)")
    ap.add_argument("--gitlab", default="", help="GitLab repository URL (https://gitlab.com/<group>/<repo>.git)")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--message", default="", help="commit message (needed when there are changes)")
    ap.add_argument("--name", default="", help="commit author name (default: git config user.name)")
    ap.add_argument("--email", default="", help="commit author e-mail (default: git config user.email)")
    ap.add_argument("--init", action="store_true", help="create the repository here if there is none")
    ap.add_argument("--check", action="store_true", help="only show what would be committed and pushed")
    ap.add_argument("--repo", default=str(REPO), help=argparse.SUPPRESS)
    ap.add_argument("--git", default="", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    repo = Path(a.repo)
    urls = {"github": a.github.strip(), "gitlab": a.gitlab.strip()}
    for k, u in urls.items():
        if u and not URL_RE.match(u):
            print(f"{k} URL not understood: {u!r}")
            return 1
    if not BRANCH_RE.match(a.branch) or ".." in a.branch:
        print(f"branch name not allowed: {a.branch!r}")
        return 1

    try:
        git = Git(repo, a.git or None)
    except RuntimeError as e:
        print(e)
        return 1
    if not git.exe:
        print("git not found: install Git for Windows (https://git-scm.com/download/win), then Check again")
        return 1
    ver = git.run("--version", quiet=True).stdout.strip()
    print(f"git: {ver} ({'Git for Windows, sign-in through the Git Credential Manager' if git.windows else 'WSL'})")

    # 1. the repository
    if not (repo / ".git").exists():
        if a.check or not a.init:
            print("no git repository in this folder yet: the first Push creates it")
            return 0 if a.check else 1
        print(f"creating the repository (branch {a.branch})")
        git.run("init", "-b", a.branch, check=True)

    # 2. remotes
    have = set(git.run("remote", quiet=True).stdout.split())
    for name, url in urls.items():
        if not url:
            continue
        if name in have:
            if git.run("remote", "get-url", name, quiet=True).stdout.strip() != url:
                git.run("remote", "set-url", name, url, check=True)
                print(f"remote {name} -> {url}")
        elif not a.check:
            git.run("remote", "add", name, url, check=True)
            print(f"remote {name} -> {url}")
    targets = [n for n in REMOTES if urls[n]]
    if not targets:
        print("no GitHub or GitLab URL set (launcher: Maintenance -> GitHub / GitLab)")
        return 1

    # 3. what goes in
    head = git.run("rev-parse", "--verify", "-q", "HEAD", quiet=True).returncode == 0
    status = git.run("status", "--porcelain", "-uall", quiet=True).stdout.splitlines()
    print(f"{len(status)} changed / new / deleted file(s) since the last commit" if head else
          f"{len(status)} file(s) for the first commit")
    for line in status[:40]:
        print("    " + line)
    if len(status) > 40:
        print(f"    ... and {len(status) - 40} more")
    if a.check:
        bad = problems_in_index(git, repo, untracked=True)
        if bad:
            print("NOT pushable as it is:")
            for b in bad:
                print("    " + b)
            return 1
        print("ready to push to " + " and ".join(f"{n} ({urls[n]})" for n in targets))
        return 0

    # 4. stage, check, commit
    git.run("add", "-A", check=True, quiet=True)
    bad = problems_in_index(git, repo)
    if bad:
        git.run("reset", "-q", quiet=True)       # back to how it was: nothing committed, nothing pushed
        print("STOPPED, nothing committed or pushed:")
        for b in bad:
            print("    " + b)
        print("Remove them from git (git rm --cached <file>) and keep them in .gitignore.")
        return 1
    staged = git.run("diff", "--cached", "--name-only", quiet=True).stdout.split()
    if staged or not head:
        if not a.message.strip():
            git.run("reset", "-q", quiet=True)
            print("a commit message is needed")
            return 1
        ident = []
        if a.name:
            ident += ["-c", f"user.name={a.name}"]
        if a.email:
            ident += ["-c", f"user.email={a.email}"]
        r = git.run(*ident, "commit", "-q", "-m", a.message.strip())
        if r.returncode != 0:
            if "tell me who you are" in (r.stdout + r.stderr).lower():
                print("set the commit author: launcher Maintenance -> GitHub / GitLab -> Name and E-mail")
            git.run("reset", "-q", quiet=True)
            return 1
        print(f"committed {len(staged)} file(s): {a.message.strip()}")
    else:
        print("nothing new to commit; pushing the last commit")

    # 5. push to each remote, both even if one fails
    failed = []
    for name in targets:
        print(f"pushing to {name} ({urls[name]}), branch {a.branch} ...", flush=True)
        r = git.run("push", "--porcelain", name, f"HEAD:refs/heads/{a.branch}")
        if r.returncode == 0:
            print(f"{name}: OK")
        else:
            failed.append(name)
            h = hint_for(r.stdout + r.stderr)
            print(f"{name}: FAILED" + (f" - {h}" if h else ""))
    if failed:
        print(f"done with errors: {', '.join(failed)} not updated (the commit is kept; push again after fixing)")
        return 1
    print("done: " + " and ".join(targets) + " are up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
