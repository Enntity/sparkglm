#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Privacy regressions use synthetic fixtures, never real credentials."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('privacy', Path(__file__).resolve().parents[1] / 'scripts/check_publication_privacy.py')
privacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(privacy)


def main():
    for octets in ((10, 20, 30, 40), (172, 16, 2, 3), (192, 168, 4, 5), (100, 64, 1, 2)):
        assert privacy.findings('archive.patch', '.'.join(map(str, octets)).encode())
    for address in (b'127.0.0.1', b'0.0.0.0', b'192.0.2.10'):
        assert not privacy.findings('example.md', address)
    example = '.'.join(map(str, (10, 0, 0, 1))).encode()
    assert not privacy.findings('.env.example', example)
    assert privacy.findings('archive.patch', example)
    assert privacy.findings('file.py', b'ghp_' + b'X' * 30)
    assert privacy.findings('file.patch', b'host' + b'.local')
    assert privacy.findings('file.patch', b'/' + b'Users/' + b'example/project')
    assert privacy.findings('model.safetensors', b'')
    assert privacy.findings('archive.patch', b'\nGIT binary ' + b'patch\n')
    assert not privacy.findings('NOTICE', b'Original Author <author@users.noreply.github.com>')
    print('Publication privacy regression tests: PASS')


if __name__ == '__main__':
    main()
