# SPDX-License-Identifier: AGPL-3.0-only
"""Version queries must work before configuration or model initialization."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


class VersionQueries(unittest.TestCase):
    def test_checkout_and_image_entrypoint_report_pin_without_configuration(self):
        release = json.loads((HERE/'release.json').read_text())
        pin = json.loads((HERE/'atlas-source.json').read_text())
        self.assertRegex(release['version'], r'^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'install').mkdir()
            shutil.copy(HERE.parent/'start.sh', root/'start.sh')
            for name in ('serve.py', 'release.json', 'atlas-source.json'):
                shutil.copy(HERE/name, root/'install'/name)
            # The version path must not even source a user's environment file.
            (root/'.env').write_text('exit 97\n')
            env = {'PATH': os.environ['PATH']}
            checkout = subprocess.check_output(['bash', str(root/'start.sh'), '--version'], env=env, text=True)
            (root/'install'/'atlas-source.json').rename(root/'install'/'source-manifest.json')
            image = subprocess.check_output(['python3', str(root/'install'/'serve.py'), '--version'], env=env, text=True)
            self.assertEqual(checkout, image)
            self.assertIn(f"{release['product']} {release['version']}\n", image)
            self.assertIn(pin['commit'], image)
            self.assertIn(pin['tree'], image)
