#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Reads the launcher's smoke completion on stdin: 17*23 must be 391 with a normal stop."""
import json
import sys

try:
    choice = json.load(sys.stdin)["choices"][0]
    text = (choice.get("message", {}).get("content") or "").strip()
    finish = choice.get("finish_reason")
except (ValueError, KeyError, IndexError, AttributeError, TypeError) as error:
    sys.exit(f"smoke failed: invalid completion response ({error})")
if text != "391" or finish != "stop":
    sys.exit(f"smoke failed: expected 391 with normal stop; got {text!r}, finish_reason={finish!r}")
print("smoke: 17*23 =", text)
