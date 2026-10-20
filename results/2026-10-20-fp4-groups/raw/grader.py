# SPDX-License-Identifier: AGPL-3.0-only
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CASES = {c["id"]: c for c in json.loads((ROOT / "cases.json").read_text())}


def unique_object(pairs):
    assert len(pairs) == len(dict(pairs)), "duplicate JSON keys"
    return dict(pairs)


def grade(case, row):
    if "error" in row:
        return {"status": "inconclusive", "reason": row["error"]}
    choice = row["response"]["choices"][0]
    text = choice["message"].get("content") or ""
    if choice["finish_reason"] != "stop":
        return {"status": "inconclusive", "reason": "finish=" + choice["finish_reason"]}
    try:
        if case["grader"] == "text":
            ok = text.strip() == case["expected"]
            detail = {"expected": case["expected"], "actual": text.strip()}
        elif case["grader"] == "json":
            actual = json.loads(text, object_pairs_hook=unique_object)
            ok = json.dumps(actual, sort_keys=True) == json.dumps(case["expected"], sort_keys=True)
            detail = {"expected": case["expected"], "actual": actual}
        else:
            source = text.strip()
            fence = re.fullmatch(r"```(?:python|py)?\s*\n([\s\S]*?)\n```", source)
            if fence:
                source = fence[1]
            assert len(source) <= 12000, "oversize code"
            result = subprocess.run(
                [sys.executable, "-I", "-S", str(ROOT / "check_code.py")],
                input=json.dumps(dict(case, source=source)), text=True,
                capture_output=True, timeout=5)
            if result.returncode:
                resource_failure = result.returncode < 0 or "MemoryError" in result.stderr
                return {"status": "inconclusive" if resource_failure else "fail",
                        "reason": "resource limit" if resource_failure else "invalid or broken function",
                        "stderr": result.stderr[-2000:], "source": source}
            detail = json.loads(result.stdout)
            ok = detail["ok"]
        return {"status": "pass" if ok else "fail", "detail": detail}
    except (ValueError, AssertionError) as error:
        return {"status": "fail", "reason": repr(error), "actual": text}
    except subprocess.TimeoutExpired:
        return {"status": "inconclusive", "reason": "code execution timeout"}


