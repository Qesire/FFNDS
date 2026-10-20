"""Run a command on one idle worker GPU with the isolated project environment."""
import argparse
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", type=int, choices=[0, 1], default=0)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        ap.error("Supply a command after --")
    remote = shlex.join(["bash", str(ROOT / "scripts/worker_exec.sh"), str(args.gpu), *command])
    return subprocess.call([
        "ssh", "-o", "ProxyCommand=none", "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3", "-o", "ControlMaster=auto",
        "-o", "ControlPersist=600", "-o", "ControlPath=/tmp/ffnds-worker-%r-%p",
        "worker", remote,
    ])


if __name__ == "__main__":
    raise SystemExit(main())
