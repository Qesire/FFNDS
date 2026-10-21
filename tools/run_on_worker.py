"""Run a command on one idle worker GPU with the isolated project environment."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
from worker_identity import ssh_arguments

SSH = ssh_arguments()


def wait_for_source_visibility():
    """NFS negative dentries/attribute caches may lag new or edited sources."""
    paths = sorted([*ROOT.glob("tools/*.py"), *ROOT.glob("config/*.json"),
                    *ROOT.glob("config/*.env"), *ROOT.glob("scripts/*.sh")])
    expected = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    probe = '''import sys,json,time,hashlib
from pathlib import Path
expected=json.load(sys.stdin)
deadline=time.monotonic()+90
while True:
    missing=[]
    for name,sha in expected.items():
        try: actual=hashlib.sha256(Path(name).read_bytes()).hexdigest()
        except FileNotFoundError: actual=None
        if actual!=sha: missing.append(name)
    if not missing:
        print(json.dumps({"status":"REMOTE_SOURCE_HASHES_READY","files":len(expected)}),flush=True)
        break
    if time.monotonic()>=deadline:
        raise SystemExit("Remote source visibility timeout: "+repr(missing))
    time.sleep(2)
'''
    subprocess.run(SSH + [shlex.join(["python3", "-c", probe])], input=json.dumps(expected), text=True, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", type=int, choices=[0, 1], default=0)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        ap.error("Supply a command after --")
    wait_for_source_visibility()
    remote = shlex.join(["bash", str(ROOT / "scripts/worker_exec.sh"), str(args.gpu), *command])
    return subprocess.call(SSH + [remote])


if __name__ == "__main__":
    raise SystemExit(main())
