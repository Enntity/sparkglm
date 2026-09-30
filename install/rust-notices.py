#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Collect the license notices of the third-party Rust crates compiled into the engine binary.

rust-notices.py WORKSPACE OUT: runs `cargo metadata` (offline, locked) on WORKSPACE, walks the
dependency graph of spark-server for the build platform, and for every crate that is not one of the
engine's own workspace crates copies its license, copying and notice files to OUT/<name>-<version>/.
OUT/INDEX.tsv lists each crate with its declared license expression and source, and says so when a
crate ships no license file (its declared license then stands alone).
"""
import json, os, shutil, subprocess, sys

NOTICE_PREFIXES = ("license", "licence", "copying", "notice", "unlicense", "copyright", "authors")

def main(ws, out):
    meta = json.loads(subprocess.check_output(
        ["cargo", "metadata", "--locked", "--offline", "--format-version", "1",
         "--filter-platform", subprocess.check_output(["rustc", "-vV"], text=True).split("host: ")[1].split()[0]],
        cwd=ws, text=True))
    pkgs = {p["id"]: p for p in meta["packages"]}
    nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}
    members = set(meta["workspace_members"])
    vendor = os.path.join(os.path.realpath(ws), "vendor") + os.sep
    root = next(i for i in members if pkgs[i]["name"] == "spark-server")
    seen, stack = set(), [root]
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        stack.extend(d["pkg"] for d in nodes[i]["deps"])
    os.makedirs(out, exist_ok=True)
    rows = []
    for i in sorted(seen, key=lambda i: (pkgs[i]["name"], pkgs[i]["version"])):
        p = pkgs[i]
        crate_dir = os.path.dirname(p["manifest_path"])
        own = i in members and not os.path.realpath(crate_dir).startswith(vendor)
        if own:
            continue
        dest = os.path.join(out, f'{p["name"]}-{p["version"]}')
        files = [f for f in sorted(os.listdir(crate_dir))
                 if f.lower().startswith(NOTICE_PREFIXES) and os.path.isfile(os.path.join(crate_dir, f))]
        if p.get("license_file"):
            lf = os.path.join(crate_dir, p["license_file"])
            if os.path.isfile(lf) and os.path.basename(lf) not in files:
                files.append(os.path.relpath(lf, crate_dir))
        os.makedirs(dest, exist_ok=True)
        for f in files:
            shutil.copyfile(os.path.join(crate_dir, f), os.path.join(dest, os.path.basename(f)))
        source = p.get("source") or ("vendored in the engine tree" if not own else "")
        rows.append((p["name"], p["version"], p.get("license") or "(see files)", source,
                     ", ".join(os.path.basename(f) for f in files) or "NO LICENSE FILE IN THE CRATE"))
    with open(os.path.join(out, "INDEX.tsv"), "w") as fh:
        fh.write("crate\tversion\tlicense\tsource\tfiles\n")
        for r in rows:
            fh.write("\t".join(r) + "\n")
    missing = sum(1 for r in rows if r[4].startswith("NO LICENSE"))
    print(f"{len(rows)} third-party crates, {missing} without a license file in the crate")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
