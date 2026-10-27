<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# SparkGLM 0.2.0 on engine c87f2790: TP2 qualification, 2026-10-27

SparkGLM 0.2.0 now pins `spark-shared` `c87f2790`, which adds
[Enntity/atlas#16](https://github.com/Enntity/atlas/pull/16) to the guard
engine of the [earlier qualification](../2026-10-27-common-engine-020/RESULT.md).
The exact image was built from this branch and served on two DGX Sparks
(03/04) in Strict, Strict native MTP and Spark. Each mode passed all 109 live
checks. The engine fixes in atlas#16 hold live:
- A forced `tool_choice` keeps the guard canary's array in schema order.
- A forced `tool_choice` enforces single-value `enum`s.
- Streamed content keeps text-spelled call markup the same way blocking
  content does.

This is a TP2 qualification. It is not a publication, a TP4 check or a
performance result.

## Image and identity

- Product commit `5f1716c`, install tree `3bf64a4d04c99223df6e50cd39f3b24dc1608515`.
- Atlas `spark-shared` `c87f27906016c874d5a356429e7816987f5b1cc6`, tree `65f088b595ecedc9b79cde411ad7e04573269b2d`: the atlas#16 merge commit on top of `d0cead14`, the engine SparkQwen 0.3.3 ships.
- Image `ghcr.io/enntity/atlas-sparkglm:3bf64a4d04c9`, ID `sha256:7cb36f010f128f215bad4e1fb86e7b1944e2b612452e7db7678c099f21ae3b03`, binary SHA-256 `131a821b8485db15a9db932a2775778966a62e4a7201f37792b941155cbf92d4`. It is a local build, not pushed.
- `install/build.sh` built it on 03 from a clean checkout of the pushed commit in 12.5 minutes. The checks stage passed 9,134 tests in 156 suites, with 0 failures, 271 ignored and full clippy on Rust 1.93.1. That includes atlas#16's new grammar and stream tests. The image was copied to 04 and has the same image ID.
- In every mode both ranks ran the same image, binary and checkpoints: FP4-Local `76348ddb` and DFlash2 `7d74cdd8`. Both ranks had the clock-cap service active. `--tool-guard=on` was set everywhere, and Strict native MTP ran with `SPARKGLM_STRICT_MTP=1`.

## Live checks

Strict and Spark used `./start.sh up` with the `4x1m` profile and passed the
`17*23 = 391` readiness check. Strict native MTP used
`install/start-node.sh --strict-mtp` on both ranks, as in the earlier run.

| Suite | Checks | Strict | Strict MTP | Spark |
|---|---|---:|---:|---:|
| Arithmetic | blocking + streaming | 2/2 | 2/2 | 2/2 |
| Literal tool XML (replay of [2026-10-22](../2026-10-22-literal-tool-xml/README.md)) | OpenAI + Anthropic, blocking + streaming, real calls | 32/32 | 32/32 | 32/32 |
| Tool-call guard (SparkQwen 0.3.3 cases) | 16 cases × 2 APIs × 2 transports + prose | 66/66 | 66/66 | 66/66 |
| Structured output | JSON schema (both transports), JSON object | 3/3 | 3/3 | 3/3 |
| Prefix cache | repeated 26K-token prompt; exact answer, 0 → 26,192 cached | 1/1 | 1/1 | 1/1 |
| Image | two blocking choices, one stream, streamed `n=2` gets the documented 400 | 3/3 | 3/3 | 3/3 |
| Short video | red then blue, both transports | 2/2 | 2/2 | 2/2 |

Each mode logged exactly 36 guard WARN events: 8 `metadata_access`, 12
`blocked_host` and 16 `credential_transfer`. These match the expected reason
for every case and contain no argument values. Generated tools were never
executed. Strict and Strict native MTP logged no other warnings or errors. Spark
logged 195 `spec-decode boundary: truncated drafts crossing grammar
transition` warnings, the same count as on the pre-merge atlas#16 image.

The first Spark start stopped at load with "No memory left for KV cache". It
began right after the Strict native MTP containers stopped, and rank 0 had
10.4 GB free at KV accounting. Once both nodes showed more than 107 GB free, a
second start on the same image passed. The failed start's log is kept in the
receipts.

## Forced tool choice and streamed markup

| Probe | 0.2.0 on `d0cead14`, Strict | Strict | Strict MTP | Spark |
|---|---:|---:|---:|---:|
| Guard canary array, `auto` / named function / `required`, byte-exact | yes / no / no | yes / yes / yes | yes / yes / yes | yes / yes / yes |
| Single-value `enum` enforced (string, array, object) | 0/3 | 3/3 | 3/3 | 3/3 |
| Malformed forced call: streamed content = blocking | no | yes | yes | yes |
| Spelled markup as text, no tools / `auto`: streamed = blocking | no / no | yes / yes | yes / yes | yes / yes |

The spelled-markup baseline ran on the earlier 0.2.0 image (`c98ddd9e55de`) on
the same pair. It returned `tool_call>probe</tool_call>` in the stream and the
full `<arg_key>city</arg_key><arg_value>Oslo</arg_value>` text when blocking.
[`docs/LIMITATIONS.md`](../../docs/LIMITATIONS.md) records the gaps that remain:
- Other argument types are still not enforced.
- A forced call can still end as spelled text, with no call delivered.

## Scope and restoration

Not covered: TP4 (it needs four Sparks), throughput, capacity, long context,
needles, RigMark and ToolEvalBench, registry publication, and managed LLooM
recipe serving. No speed claim follows from these timings.

The pair was held through LLooM (`r-d04a7421`, job `j-3d2ea732`). Release
restored production Qwen on 03/04, and the gateway answered `42`
(`qwen3.8-flash-next-atlas`, `stop`) at 2026-10-27T21:28:11Z.

`qualification.json` has the per-mode counts, identities, guard log summaries
and probe results. `receipts.tar.gz` holds:
- the harness, fixtures, launcher and `.env`;
- the build log;
- every raw response;
- per-mode start logs, both ranks' logs, identity, guard summaries and probes;
- the failed Spark start;
- the 0.2.0 spelled-markup baseline.

`SHA256SUMS` covers these files.
