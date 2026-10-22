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
    # Keep expected_hostname as the immutable historical source/redaction
    # origin. A separately configured execution host permits a byte-verified
    # cache mirror after an explicit user-directed worker migration.
    config = json.loads(private_config_path().read_text())
    execution = config.get('execution_hostname', expected_hostname())
    if not isinstance(execution, str) or not execution.strip() or any(c.isspace() for c in execution):
        raise RuntimeError('Invalid privately configured execution_hostname')
    if socket.gethostname().split('.')[0] != execution.split('.')[0]:
        raise RuntimeError('This operation requires the privately configured GPU worker')


def matches_worker(recorded_host):
    """Accept the verified manifest origin or the actual configured executor.

    This checks provenance only. restored_assets still verifies exact source,
    checkpoint, cache, sizes and hardlink identities on the execution node.
    """
    require_worker()
    return (isinstance(recorded_host, str) and recorded_host.split('.')[0]
            in ('worker', expected_hostname(), socket.gethostname().split('.')[0]))


def ssh_arguments():
    config = Path(os.environ.get('FFNDS_SSH_CONFIG',
        str(Path(os.environ.get('XDG_CONFIG_HOME', str(Path.home() / '.config'))) / 'ffnds/ssh_config')))
    return ['ssh', '-F', str(config), '-o', 'ProxyCommand=none', '-o', 'BatchMode=yes',
            '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3',
            '-o', 'ControlMaster=auto', '-o', 'ControlPersist=600',
            '-o', 'ControlPath=/tmp/ffnds-worker-%h-%r-%p', 'ffnds-worker']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', required=True)
    parser.parse_args()
    require_worker()
