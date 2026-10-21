"""Keep connection details private while enforcing the configured worker host."""
import argparse
import json
import os
from pathlib import Path
import socket


def private_config_path():
    return Path(os.environ.get('FFNDS_WORKER_CONFIG',
        str(Path(os.environ.get('XDG_CONFIG_HOME', str(Path.home() / '.config'))) / 'ffnds/worker.json')))


def expected_hostname():
    value = json.loads(private_config_path().read_text()).get('expected_hostname')
    if not isinstance(value, str) or not value.strip() or any(c.isspace() for c in value):
        raise RuntimeError('Private worker configuration requires one expected_hostname')
    return value.split('.')[0]


def require_worker():
    if socket.gethostname().split('.')[0] != expected_hostname():
        raise RuntimeError('This operation requires the privately configured GPU worker')


def matches_worker(recorded_host):
    """Accept an original private manifest or a neutral public worker label."""
    require_worker()
    return recorded_host in ('worker', socket.gethostname().split('.')[0], socket.gethostname())


def ssh_arguments():
    config = Path(os.environ.get('FFNDS_SSH_CONFIG',
        str(Path(os.environ.get('XDG_CONFIG_HOME', str(Path.home() / '.config'))) / 'ffnds/ssh_config')))
    return ['ssh', '-F', str(config), '-o', 'ProxyCommand=none', '-o', 'BatchMode=yes',
            '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3',
            '-o', 'ControlMaster=auto', '-o', 'ControlPersist=600',
            '-o', 'ControlPath=/tmp/ffnds-worker-%r-%p', 'ffnds-worker']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', required=True)
    parser.parse_args()
    require_worker()
