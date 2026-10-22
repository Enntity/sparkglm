<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Short-prose loop suppression qualification

Atlas PR7 removes tool steering based only on short assistant prose in request history. The separate signature/result-based repeated-call protection remains. SparkGLM PR57 permits the exact diagnostic escape hatch, but this qualification did not set it.

The exact pinned engine `e2fcc5cd4e57421b4828f142957f82bbf10aee52` passed all 15 short ToolEvalBench scenarios at each requested pressure level: 10%, 33%, 57%, and 80% of a configured 131,072-token context. This includes the original TC-07 report/read/contacts/email workflow at every level. Tools use the benchmark's simulated fixtures; no real email is sent.

`qualification.json` records the candidate, hardware/profile, and unit-test results. `sweep.json` is the completed sweep record exported from ToolEvalBench's SQLite store. Version 2.0.6 does not write a sweep to `--json-file`; its SQLite result is authoritative. The pressure percentages are the benchmark's requested levels; its `fill_tokens` fields are retained without reinterpreting them as actual tokenizer counts.

```sh
tool-eval-bench --base-url http://127.0.0.1:18893/v1 \
  --model glm-5.3-flash-atlas --backend unknown --short \
  --context-pressure-sweep 0.1-0.8 --sweep-steps 4 \
  --context-size 131072 --seed 42 --temperature 0 --no-think \
  --max-turns 8 --timeout 180 --no-live --no-probe-engine
```

The package was installed from upstream tag v2.0.6, commit `c3868bff099592c9a1045de2c9a3dc24abebb7fb`. The GLM template forces thinking on despite `--no-think`; this is recorded rather than treating the flag as effective.

The native server suite passed 2,843 tests with 13 ignored. All 3,900 tracked build-source files were checked against the pinned source before compiling. The candidate image ID is local build evidence, not a published registry digest or proof of a managed-serving deployment. This is a correctness qualification, not a performance comparison.

Atlas's broader CI remains red for the limitations recorded on PR7: CUDA stub linking, pre-existing model/comm lint failures, spelling/file-size findings, locked dependency advisories, and the CLA check. Formatting, SPDX, kernel compilation, ancestry, provenance and threading checks pass.
