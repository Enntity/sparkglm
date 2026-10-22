<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# Literal tool XML qualification

[Atlas PR8](https://github.com/Enntity/atlas/pull/8) fixes literal tool markup being interpreted as protocol control, grammar activation inside quoted examples, and streamed content loss. The failures were reproduced directly against Atlas without Zcode.

The exact pinned engine `3bc73e0b850435f8d4519ab998a85ef9f7d82706` passed all 28 live controls: OpenAI and Anthropic, blocking and streaming, with inline code, fenced full calls, quoted tags, bare tags, no-tools requests, malformed examples and actual tool calls. `canaries.json` records the synthetic requests and returned content. Exact-content checks include whitespace and delimiters; real-call controls require one call with value 42. No real tools execute in this matrix.

ToolEvalBench 2.0.6 passed all 15 short scenarios (30/30 points) on the same native image. `tool-smoke.json` retains scenario outcomes without the benchmark prompts or transcripts. Its tools are simulated fixtures; no real messages are sent. The earlier 60/60 pressure sweep in `../2026-10-22-tool-boundaries` qualified the preceding loop-fix revision, not this revision.

```sh
tool-eval-bench --base-url http://127.0.0.1:18893/v1 \
  --model glm-5.3-flash-atlas --backend vllm --short \
  --context-size 131072 --seed 42 --temperature 0 --no-think \
  --max-turns 8 --timeout 180 --no-live --no-probe-engine \
  --json-file xml-tool-smoke-3bc73e0b.json
```

The `vllm` flag selects the benchmark's compatible client adapter; Atlas is the actual engine. Version 2.0.6 rejects `unknown` on this non-sweep command before issuing requests. The checkpoint template forces thinking despite the no-think request.

The native server suite passed 2,889 tests and the local suite passed 2,881, each with 13 ignored. All 3,906 tracked build-source hashes were verified before compilation. Formatting, license headers and diff checks pass. Broader Atlas CI has inherited failures; local workspace Clippy is blocked by unchanged atlas-plugin lints, and typos is unavailable locally. See the PR for CI findings.

This is a correctness receipt, not a performance comparison. The local image ID records the tested binary; it is not a published registry digest or proof of a managed-serving deployment.
