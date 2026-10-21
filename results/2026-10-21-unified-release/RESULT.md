<!-- SPDX-License-Identifier: AGPL-3.0-only -->
# SparkGLM unified TP2 / TP4 release — 2026-10-21

One image supplies Strict and Spark modes on TP2 and TP4, prefix caching,
optional Strict native MTP, and explicit switchless chain/ring support on TP4.
The published image is deployed through LLooM on ennspark01/02 in Spark mode,
with normal clock caps restored. Both ranks match the release runtime hashes;
the gateway returns `42`. Qwen on 03/04 and the Presence route were verified.

## Release identity

- Product release: [`unified-20261021`](https://github.com/Enntity/sparkglm/releases/tag/unified-20261021).
- Image: `ghcr.io/enntity/atlas-sparkglm:e3792e6e8c51`.
- Immutable pull: `ghcr.io/enntity/atlas-sparkglm@sha256:0d616370f36f36c8f152a4fcfc68839ab4d12f8a532917ad7b4df71f41f0d010`.
- Install tree: `e3792e6e8c515dec593e674fc2e022fc83315048`.
- Public Atlas source: [`sparkglm/atlas-20261021-unified`](https://github.com/Enntity/atlas/tree/sparkglm/atlas-20261021-unified), commit `f4d95a100516338c938a4524e23a65a2c4c3206e`, tree `58ed69bb21acc6bec5aa3cf0f2f339dc2366d0a2`.
- Runtime executable SHA-256: `e79d163dd10c07e3244de7c26a73c915a35ae3df08d5db34162243baaf182467`.

The accepted source is one integration commit on the previous published GLM
foundation. Its tree matches the qualified `d3620ffb` tree except for an
example address in a source comment. The image retains all qualified RC
layers, executable bytes, native libraries, launcher and four profile files;
it adds only the public source manifest and build provenance. This is a
metadata-only repack, not a claim of a new cold build. Source mapping and
per-file hashes are preserved here. Shared Atlas and SparkQwen branches were
not changed by this release.

Both Sparks pulled the exact digest anonymously. A fresh anonymous clone of
the public product release branch ran `./start.sh` against the pinned existing
checkpoints, launched Spark mode, and passed the `17*23 = 391` smoke test.
Checkpoint downloads were reused after checking their recorded revisions;
this did not repeat a full download. Product install bytes are identical to
those in the final release; later product changes add docs, art and receipts.

## Current quick decode measurements

GPU clocks uncapped, two runs per workload, low reasoning effort, 4,096-token
output allowance. All six TP2 responses stopped naturally and passed the
harness completion checks. These are short single-stream observations, not a
new broad benchmark campaign or a statistical significance claim.

| Workload | TP2 median tok/s | TP4 chain median tok/s | TP4 ring median tok/s |
|---|---:|---:|---:|
| Prose | 37.489 | 57.145 | 58.328 |
| Code | 67.166 | 104.147 | 106.368 |
| Structured | 89.752 | 137.936 | 141.363 |

TP2 ran the published image; TP4 ran the byte-identical qualified runtime.
The same RigMark-derived prompts, nonces, sampling settings and timing code
were used. TP2 and TP4 can produce different outputs; this is not a token-exact
cross-TP comparison. Chain versus ring preserves each output, reasoning and
stream hash, with median gains of 2.07%, 2.13% and 2.48% respectively.
Every TP4 repetition is in the [qualification report](../2026-10-21-unified-rc/RESULT.md#chain-versus-ring--spark-tp4).

Every TP2 repetition:

| Workload | Run | tok/s | Completion tokens | Finish |
|---|---:|---:|---:|---|
| prose | 1 | 36.994 | 1043 | stop |
| prose | 2 | 37.985 | 925 | stop |
| code | 1 | 68.380 | 2140 | stop |
| code | 2 | 65.952 | 1573 | stop |
| structured | 1 | 89.884 | 441 | stop |
| structured | 2 | 89.621 | 441 | stop |

`tp2-release-speed.json` retains exact payloads, completions, timing and hashes;
`raw/` contains the harness and source fixtures. The arcade cards show these
medians rounded to one decimal place, not selected maximum runs. Serving
clock caps were restored after measurement; these numbers do not describe
clock-capped live serving.

## Qualification retained and extended

The [unified RC qualification](../2026-10-21-unified-rc/RESULT.md) and its
725-file archive cover final TP2 Spark and Strict-native checks, TP4 Spark on
chain and ring, Strict-native ring, C4 arithmetic, JSON/tools, cancellation,
slot reuse, the original long retrieval cases, exact kernel oracles,
near-limit native concurrency and two forced NVMe restore cycles.

The [earlier full TP2 qualification](../2026-10-20-tp2-qualification/RESULT.md)
and its 14,828-file archive retain the broader tests and their original
identities, including failures. Strict/Spark scored 30/30 behavior, 14/14
tools, 12/12 short retrieval and 60/60 long retrieval; prompt NLL on 25,050
tokens was 0.959750/0.962332. Its Spark arithmetic result was 39/40 versus
Strict 40/40. Turning off shared-expert TP splitting fixed the affected
arithmetic case in subsequent matched controls and final C4 checks. The
original report remains a failure record rather than being rewritten.

Native-MTP sampling's raw-argmax shortcut is now greedy-only. The retained
960-draw native and 960-draw serial comparison detected no distribution
difference; that limited test does not prove distribution equivalence.
Near-1M admission/headroom fixes passed the affected follow-ups. TP4 final
capacity evidence uses four branches sharing a 1,046,988-token prefix, not
four independent million-token prompts. Strict retains FP8 KV and the
engine's prefill activation policy; no cross-engine bitwise claim is made.

The earlier Kindling comparison is in the README and retained TP2 report:
80 measured runs and 193 streams per mode including warmups, `8x128k`,
uncapped clocks, explicit zero thinking budget and 512/768-token output
limits. Spark's shared-expert split was already off in that profile. Those
results are reused with attribution; Kindling was not rerun for publication.

The final product passed 56 launcher tests, three converter tests, shellcheck,
Python compilation and receipt checksum verification. The qualification
report records focused Linux/CUDA/runtime checks. This release does not
claim a clean full-workspace engine Clippy run or repeat a cold source build.

## Receipt publication

`SHA256SUMS` covers all files here. The two linked qualification directories
retain raw request/response data, failed controls, all repetitions, source
patches, per-rank startup and hardware receipts. The older TP2 archive projects
20 Docker inspection files to container/image/state fields to omit unrelated
host configuration; its publication note and original archive hash document
that projection. Benchmark outputs and timings are unchanged. Private gateway
inventories, authentication and configuration backups are excluded.

The release source package contains exact public product and engine snapshots,
all result bundles, licenses and a cold-build script. The script was syntax
checked; the published image's incremental runtime build and live tests are
the execution evidence. Container and model identities should be checked
against these receipts when reproducing results.
