"""Fetch only the pinned Menagerie Panda model and preserve upstream licenses."""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "assets/menagerie"
SOURCE = "https://github.com/google-deepmind/mujoco_menagerie.git"
REVISION = "8161bba264d7fa7c99ca301e91e7fb44737676ad"


def git(*args):
    return subprocess.run(
        ["git", "-C", str(DEST), *args], check=True, text=True, capture_output=True
    ).stdout.strip()


def main():
    if DEST.exists():
        if not (DEST / ".git").exists():
            raise SystemExit(
                f"Refusing to overwrite existing non-Git directory: {DEST}"
            )
        if git("remote", "get-url", "origin") != SOURCE:
            raise SystemExit(
                f"Unexpected origin in {DEST}; leave it intact and inspect manually"
            )
        if git("status", "--porcelain"):
            raise SystemExit(
                f"Local asset changes in {DEST}; leave them intact and inspect manually"
            )
        if (
            git("rev-parse", "HEAD") == REVISION
            and (DEST / "franka_emika_panda/scene.xml").exists()
        ):
            print(f"Panda assets already installed at {REVISION}")
            return
    else:
        DEST.mkdir(parents=True)
        git("init")
        git("remote", "add", "origin", SOURCE)
    git("config", "remote.origin.promisor", "true")
    git("config", "remote.origin.partialclonefilter", "blob:none")
    git("sparse-checkout", "init", "--cone")
    git("sparse-checkout", "set", "franka_emika_panda")
    git("fetch", "--depth", "1", "--filter=blob:none", "origin", REVISION)
    git("checkout", "--detach", REVISION)
    print(f"Installed Panda assets at {REVISION}: {DEST}")


if __name__ == "__main__":
    main()
