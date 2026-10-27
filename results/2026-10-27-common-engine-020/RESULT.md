<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# SparkGLM 0.2.0 on the common engine: TP2 qualification, 2026-10-27

The exact 0.2.0 image was built from this branch and served on two DGX Sparks
(03/04) in Strict, Strict native MTP and Spark modes. Each mode passed
105 of 109 live checks. All guard blocks, literal-XML replays, structured
output, prefix reuse, image, video and readiness checks passed. The four misses
are one guard-canary case. The guard correctly delivered that call, but the
model put the array in a different order under forced `tool_choice`. The 0.1.0
release does the same, so this is not a 0.2.0 regression. This is a TP2
qualification. It is not a publication, a TP4 check or a performance result.

## Image and identity

- Product commit `03814c6a`, install tree `c98ddd9e55de889b7c28bbcfc40a50b83214b33e`.
- Atlas `spark-shared` `d0cead146eedbf8848c7f5da49f83e183228830c`, tree `c8ee0b9505afb2680fa21dc40bf14c872b8879c1`, the engine SparkQwen 0.3.3 ships.
- Image `ghcr.io/enntity/atlas-sparkglm:c98ddd9e55de`, ID `sha256:6cc8f6e00038acac853c601b41a06946e1d753466f91f87f9745d451df5834f7`, binary SHA-256 `537d4c6f6de2ee917923b085adc1c399b5898585b9df2a7f1edd4594f96c1d21`. It is a local build, not pushed.
- `install/build.sh` built it on 03 from a fresh clone in 16 minutes. The checks stage passed 9,125 tests in 156 suites, with 0 failures, 271 ignored and full clippy on Rust 1.93.1. These include the guard's GLM boundary tests in `spark-server`. The image was copied to 04 and has the same image ID and layers.
- In every mode both ranks ran the same image, binary and checkpoints: FP4-Local `76348ddb` and DFlash2 `7d74cdd8`. Both ranks had the clock-cap service active. Each mode's `identity.json` records the engine argv, which includes `--tool-guard=on --tool-guard-max-bytes=1048576`.

## Live checks

Strict and Spark used `./start.sh up` with the `4x1m` profile. Strict native
MTP used `install/start-node.sh --strict-mtp` on both ranks with the same
arguments, because `start.sh` has no option for it. Both `start.sh` launches
passed the new readiness check: `17*23 = 391` with a normal stop.

| Suite | Checks | Strict | Strict MTP | Spark |
|---|---|---:|---:|---:|
| Arithmetic | blocking + streaming | 2/2 | 2/2 | 2/2 |
| Literal tool XML (replay of [2026-10-22](../2026-10-22-literal-tool-xml/README.md)) | OpenAI + Anthropic, blocking + streaming, real calls | 32/32 | 32/32 | 32/32 |
| Tool-call guard (SparkQwen 0.3.3 cases) | 16 cases × 2 APIs × 2 transports + prose | 62/66 | 62/66 | 62/66 |
| Structured output | JSON schema (both transports), JSON object | 3/3 | 3/3 | 3/3 |
| Prefix cache | repeated 26K-token prompt; exact answer, 0 → 26,192 cached | 1/1 | 1/1 | 1/1 |
| Image | two blocking choices, one stream, streamed `n=2` gets the documented 400 | 3/3 | 3/3 | 3/3 |
| Short video | red then blue, both transports | 2/2 | 2/2 | 2/2 |

The 18 blocked cases were blocked over both APIs and both transports. Chat
Completions returned `content_filter` / `guard_stop: "tool_call_guard"` and
Anthropic returned `stop_reason: "refusal"`, with the fixed refusal and no tool
header or argument fragment. Each mode logged exactly 36 guard WARN events:
8 `metadata_access`, 12 `blocked_host` and 16 `credential_transfer`. These
match the expected reason for every case and contain no argument values.
Generated tools were never executed and no destination in an argument was
contacted.

In the first Strict run, the image suite asked for a streamed `n=2` response.
The engine rejects that by design ("n > 1 is not supported in streaming mode").
The corrected suite passed 3/3 on the same running server.

## Findings

**Forced `tool_choice` does not enforce `enum`, and it re-serializes arguments.**
All four misses are `independent-operations`. With `tool_choice` naming the
function, or `required`, the model wrote the two operations in reverse order.
The guard delivered the call unchanged, which is the expected guard outcome.
With `tool_choice: "auto"` the same request returns the arguments
byte-for-byte in schema order. A raw `/v1/completions` replay of the checkpoint
template, which bypasses the parser and guard, also produces schema order.
Probes show that forced choice accepts values outside a single-value `enum`
(strings, arrays and objects). They also show that delivered arguments have
their object keys sorted. 0.1.0 (Atlas `095ff223`) does exactly the same.
Spark preserved the order in the separate probe run, so the order depends on
the decoding path.

**Streamed content drops malformed tool markup.** Under forced choice with
values that conflict with the `enum`, the model can write `tool_call>…` as text
without the native opener. Blocking returns all of that text. 0.2.0's stream
keeps only `tool_call>probe</tool_call>` and drops the `<arg_key>…</arg_value>`
middle. No call is delivered in either transport. 0.1.0's stream is worse here:
it is empty and ends at the token limit. Both findings are engine behavior in
`spark-shared`, not part of this product change.

## Control: 0.1.0 on the same pair

The 0.1.0 release ran in Strict on the same pair:
`ghcr.io/enntity/atlas-sparkglm:33a8db7e0f4a`, image
`sha256:cf5dfc323bbe…`. It reproduced both findings. It has no guard, so all
36 blocked-case rows were delivered as calls, as expected. It passed 30 of 32
literal-XML replays: the two streamed `malformed-known` cases returned no
content and hit the token limit. 0.2.0 passes all 32 in every mode.

## Scope and restoration

Not covered: TP4 (it needs four Sparks), throughput, capacity, long context,
needles, RigMark and ToolEvalBench, registry publication, and managed LLooM
recipe serving. No speed claim follows from these timings.

The pair was held through LLooM (`r-5f922fb0`, job `j-350e74e5`). Release
restored production Qwen on 03/04, and the gateway answered `42`
(`qwen3.8-flash-next-atlas`, `stop`) at 2026-10-27T14:17:52Z.

`qualification.json` has the per-mode counts, identities, guard log summaries
and probe results. `receipts.tar.gz` holds:

- the harness, fixtures, launcher and `.env`;
- the build log;
- every raw response;
- per-mode start logs, rank identity, guard event lines and probes;
- the 0.1.0 control.

`SHA256SUMS` covers these files.
