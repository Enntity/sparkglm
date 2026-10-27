# SPDX-License-Identifier: AGPL-3.0-only
"""start.sh refuses to serve until both Sparks run gpu-clock-cap.service.

Kept outside install/ so these host-only checks don't change the image tag.
"""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fake_tree(root, here_active, worker_active):
    """A start.sh whose fabric resolves, with stub systemctl, ssh and docker."""
    text = (ROOT/'start.sh').read_text()
    start = text.index('fabric_ports() {')
    end = text.index('\npick_fabric()', start)
    text = text[:start]+"fabric_ports() { printf '%s\\n' 'left0 nic0 192.0.2.1/24'; }\n"+text[end:]
    (root/'start.sh').write_text(text)
    shutil.copy(ROOT/'fabric.py', root/'fabric.py')
    (root/'install').mkdir()
    shutil.copy(ROOT/'install'/'serve.py', root/'install'/'serve.py')
    bindir = root/'bin'
    bindir.mkdir()
    stubs = {
        'systemctl': f'exit {0 if here_active else 3}',
        # The worker answers the fabric inventory and the systemctl probe.
        'ssh': 'case "$*" in *systemctl*) exit %d;; esac\n'
               "printf '%%s\\n' 'right0 net0 192.0.2.2/24'" % (0 if worker_active else 3),
        'docker': f'touch {root}/DOCKER_CALLED',
        # Never let a passing check reach the real download, copy or health wait.
        'hf': 'exit 1', 'rsync': 'exit 1', 'curl': 'exit 1',
    }
    for name, body in stubs.items():
        p = bindir/name
        p.write_text('#!/bin/sh\n'+body+'\n')
        p.chmod(0o755)
    # No IMAGE: right after the check, start.sh asks the absent install/build.sh
    # for the tag and stops there.
    env = dict(os.environ, PATH=f'{bindir}:{os.environ["PATH"]}', WORKER='test-peer',
               MODEL_ROOT=str(root/'models'))
    for key in list(env):
        if 'FABRIC' in key or key in ('LEADER_ADDRESS', 'CLOCK_CAP', 'IMAGE'):
            env.pop(key)
    return env


class ClockCapCheck(unittest.TestCase):
    def run_up(self, here_active, worker_active, **extra):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = dict(fake_tree(root, here_active, worker_active), **extra)
            result = subprocess.run(['bash', str(root/'start.sh'), 'up'], env=env,
                                    capture_output=True, text=True, timeout=30)
            return result, (root/'DOCKER_CALLED').exists(), (root/'models').exists()

    def test_uncapped_host_refuses_before_docker_and_models(self):
        for here, worker, name in [(False, True, 'here'), (True, False, 'test-peer')]:
            with self.subTest(uncapped=name):
                result, docker, models = self.run_up(here, worker)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn(f'not capped on {name}', result.stderr)
                self.assertIn('./start.sh clock-cap', result.stderr)
                self.assertFalse(docker)
                self.assertFalse(models)

    def test_capped_or_skipped_passes_the_check(self):
        for here, worker, extra in [(True, True, {}), (False, False, {'CLOCK_CAP': '0'})]:
            with self.subTest(here=here, worker=worker, **extra):
                result, docker, _ = self.run_up(here, worker, **extra)
                self.assertNotIn('not capped', result.stderr)
                self.assertIn('build.sh', result.stderr)  # went on to the next step
                self.assertFalse(docker)


class ClockCapCommand(unittest.TestCase):
    def test_cap_must_be_a_number_before_any_sudo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = fake_tree(root, True, True)
            for name in ('sudo', 'ssh'):
                p = root/'bin'/name
                p.write_text(f'#!/bin/sh\ntouch {root}/{name.upper()}_CALLED\n')
                p.chmod(0o755)
            env['GPU_CLOCK_MAX_MHZ'] = '2200; touch /tmp/pwned'
            result = subprocess.run(['bash', str(root/'start.sh'), 'clock-cap'], env=env,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn('must be a number', result.stderr)
            self.assertFalse((root/'SUDO_CALLED').exists())
            self.assertFalse((root/'SSH_CALLED').exists())


if __name__ == '__main__':
    unittest.main()
