# SPDX-License-Identifier: AGPL-3.0-only
"""Execute only restricted pure-Python function answers in a bounded child."""
import ast
import copy
import json
import resource
import sys

assert sys.platform.startswith("linux"), "Memory-bounded runner requires Linux"
resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
resource.setrlimit(resource.RLIMIT_DATA, (256 * 1024**2, 256 * 1024**2))
case = json.load(sys.stdin)
tree = ast.parse(case["source"])
assert len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef), "expected one function"
fn = tree.body[0]
assert fn.name == case["function"] and not fn.decorator_list and fn.returns is None
assert all(a.annotation is None for a in fn.args.args)
builtins = {k: v for k, v in vars(__builtins__).items()
            if k in {"len", "range", "enumerate", "sorted", "min", "max", "zip",
                     "reversed", "abs", "str", "int", "bool", "list", "tuple",
                     "dict", "set", "sum", "any", "all"}}
methods = {"append", "extend", "copy", "sort", "pop", "items", "values", "keys"}
for node in ast.walk(tree):
    assert not isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.Global,
                                 ast.Nonlocal, ast.With, ast.AsyncWith, ast.AsyncFunctionDef,
                                 ast.Await, ast.Delete)), "unsupported statement"
    if isinstance(node, ast.Name):
        assert "__" not in node.id, "dunder name"
    if isinstance(node, ast.Attribute):
        assert node.attr in methods, "unsupported attribute"
    if isinstance(node, ast.Call):
        assert (isinstance(node.func, ast.Name) and
                node.func.id in {*builtins, fn.name}) or (
                    isinstance(node.func, ast.Attribute) and node.func.attr in methods
                ), "unsupported call"
namespace = {"__builtins__": builtins}
exec(compile(tree, "<model-function>", "exec"), namespace)
results = []
for test in case["tests"]:
    argument = copy.deepcopy(test["input"])
    answer = namespace[fn.name](argument)
    results.append({"ok": answer == test["expected"] and argument == test["input"],
                    "actual": answer, "expected": test["expected"],
                    "input_unchanged": argument == test["input"]})
print(json.dumps({"ok": all(x["ok"] for x in results), "tests": results}))
