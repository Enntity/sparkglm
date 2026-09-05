#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Scan tracked content, or all reachable history, without printing secrets.

This bounded detector is a gate, not a guarantee that arbitrary secrets or
personal information can be recognized. Human release review remains required.
"""

import argparse
import ipaddress
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IP = re.compile(rb"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
NETWORKS = [ipaddress.ip_network((base, bits)) for base, bits in (
    (0x0A000000, 8), (0xAC100000, 12), (0xC0A80000, 16), (0x64400000, 10)
)]
# The original serving kit's documented example network, not lab addresses.
EXAMPLE_IPS = {ipaddress.ip_address(0x0A000000 + n) for n in range(1, 5)}
EXAMPLE_PATHS = {
    '.env.example', '.env.tp4.example', 'start.sh', 'start-tp4.sh',
    'docs/upstream/MIA_RECIPE_README.md', 'tests/test_numeric_config.py',
}
SECRET = re.compile(
    rb'BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY|'
    rb'gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|'
    rb'hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|'
    rb'xox[baprs]-[A-Za-z0-9-]{15,}|AIza[A-Za-z0-9_-]{30,}'
)
LOCAL = re.compile(rb'/' rb'Users/[^/\s]+|\b[A-Za-z0-9_-]+\.local\b')
PROHIBITED = re.compile(
    r'(^|/)(\.env|\.DS_Store)$|'
    r'\.(pt|safetensors|gguf|bin|so|pyc|key|pem|p12|mp4|mov|mkv|sqlite|db)$', re.I
)


def findings(path, data):
    errors = []
    if PROHIBITED.search(path):
        errors.append('prohibited artifact filename')
    if SECRET.search(data):
        errors.append('credential-like material')
    if path != 'scripts/publication-audit.sh' and LOCAL.search(data):
        errors.append('workstation path or mDNS hostname')
    if b'\nGIT binary patch\n' in data:
        errors.append('embedded binary patch')
    for literal in set(IP.findall(data)):
        try:
            address = ipaddress.ip_address(literal.decode())
        except ValueError:
            continue
        if any(address in network for network in NETWORKS):
            if address not in EXAMPLE_IPS or path not in EXAMPLE_PATHS:
                errors.append('private or CGNAT address; use explicit configuration')
                break
    return errors


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def scan(history=False):
    failed = 0
    count = 0
    if history:
        paths = {}
        commits = git('rev-list', '--all').decode().splitlines()
        for commit in commits:
            for entry in git('ls-tree', '-rz', commit).split(b'\0'):
                if not entry:
                    continue
                meta, path = entry.split(b'\t', 1)
                _, kind, oid = meta.decode().split()
                if kind == 'blob':
                    paths.setdefault(oid, set()).add(path.decode())
        with subprocess.Popen(['git', 'cat-file', '--batch'], cwd=ROOT,
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE) as proc:
            for oid in list(paths) + commits:
                proc.stdin.write((oid + '\n').encode())
                proc.stdin.flush()
                size = int(proc.stdout.readline().split()[2])
                data = proc.stdout.read(size)
                proc.stdout.read(1)
                count += 1
                for path in paths.get(oid, {'commit metadata'}):
                    for error in findings(path, data):
                        print(f'FAIL {oid[:12]} {path}: {error}')
                        failed += 1
            proc.stdin.close()
    else:
        for path in git('ls-files', '-z').decode().split('\0'):
            if not path or not (ROOT / path).is_file():
                continue
            count += 1
            for error in findings(path, (ROOT / path).read_bytes()):
                print(f'FAIL {path}: {error}')
                failed += 1
    print(f'Publication privacy: {"FAIL" if failed else "PASS"} ({count} objects)')
    return bool(failed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', action='store_true')
    raise SystemExit(scan(parser.parse_args().history))
