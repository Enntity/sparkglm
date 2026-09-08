# NVFP4 TP2 candidate: copied Humming / MXFP8 Spark recipe

Prepared on 2026-09-06. **Unqualified experiment, disabled by default.**
No Spark was contacted, no checkpoint downloaded, and no image was built or
deployed during preparation. This is a source candidate for later hardware
qualification, not a measured replacement for EXL3.

September 7 hardware follow-up: the inherited DFlash loader ignores the
independent draft TP1 flag and keeps the target TP2 group. LLooM now rejects
that misleading request. The installed Humming input schema also falls back
to FP8 activations, so this copied profile is not proof of native FP4 compute.
The separately tested current compressed-tensors path uses actual draft TP2;
see the [tuned native NVFP4 results](../../../results/candidates/2026-09-07-nvfp4-mxfp8-1k/RESULT.md).

## Selection

The starting point is
[loud1990's dual-Spark recipe](https://github.com/loud1990/GLM-5.3-Flash-NVFP4-MXFP8-2x-DGX-Sparks/tree/d00b2ffa70e0ccbfd582088ab32448b01bfe9c67),
specifically its Jovian-derived Humming profile. Its publisher retains actual
ARM64/GB10 TP2 concurrency receipts and a cold-prefill sweep. It shares the
Mia/vLLM foundation already used by SparkGLM. This is the strongest directly
reusable starting point found for this experiment; the heterogeneous public
tests do not establish a universal fastest implementation.

Alternatives inspected at immutable revisions:

| Source | Revision | Decision |
| --- | --- | --- |
| [Tony DFlash2](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark) | `050081dc41ce6edd4d3f15fa19dc3410ba4210e3` | Keep RedHat/compressed-tensors + Marlin as an alternate if candidate quality fails; newer checkpoint choice addresses a reported ModelOpt corruption issue. Do not transfer old LibertAI benchmark figures to RedHat. |
| [eugr Spark B12X](https://github.com/eugr/spark-vllm-docker) | `6e221e8e520547394a0ad5c1dfcff694b7949005` | Native Spark recipe exists, but changes checkpoint, MTP and runtime together. |
| [amasu cluster](https://github.com/amasu/glm53-flash-cluster) | `0ab7ca7cb1067d4fcece9d525e5d90a9bbe33773` | Useful semantic/operational comparison; its September 6 log rejects the eugr alternative on its own workload. Keep that conclusion scoped to its configurations. |
| [ormandj SGLang](https://github.com/ormandj/sglang-glm53-flash-sm120) | `3797e9eeda0976a9f47d05b5d595ffc1a2ac538d` | Targets x86/SM120 RTX PRO 6000; not a drop-in two-Spark image. |
| [Mia NVFP4](https://github.com/MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark) | `aed98a13ca75140d2691cc5c651ea5817d9a3e44` | Older Ray/MTP foundation; prefer the already-tested mp/DFlash2 derivative. |

The selected recipe's source tables are
[concurrency and prefill](https://github.com/loud1990/GLM-5.3-Flash-NVFP4-MXFP8-2x-DGX-Sparks/blob/d00b2ffa70e0ccbfd582088ab32448b01bfe9c67/benchmarks/all-benchmark-results-20260828.md)
and [compatibility limits](https://github.com/loud1990/GLM-5.3-Flash-NVFP4-MXFP8-2x-DGX-Sparks/blob/d00b2ffa70e0ccbfd582088ab32448b01bfe9c67/docs/jovian-dual-gb10.md).
Their short counting-prompt results are not real-agent throughput guarantees.

## What was copied

- `overlay/` contains the upstream MXFP8 DFlash2 model and patch installer,
  copied verbatim; no new quantizer or GPU kernel was written.
- `head.sh` and `worker.sh` are extracted from upstream's actual inner shell
  scripts. The head API bind now defaults to loopback.
- `Dockerfile` is upstream's small derivative, with the formerly floating Mia
  base pinned by digest and the rank entrypoints included.
- `candidate.json` retains the Humming profile: TP2, FP8 KV, MXFP8 DFlash2 K7
  with a requested draft TP1 flag, full/piecewise graphs, four sequences, 1,024 batch tokens,
  262,144 context, utilization 0.88, text-only. Checkpoint auto-detection is
  deliberate: the selected August 28 target declares ModelOpt mixed precision.
  Do not force `modelopt_fp4` or `exl3`.

The image digest was resolved from public GHCR on the preparation date. The
target is pinned to `378ca54585c46542bad1f3cb3ed0d73ae51cdb62`, the latest
publisher commit on August 28, when upstream ran its tests. Direct config
comparison found that current `46aaae8a82032f77100f2f03e9cc11b391df3b4d`
changes `quant_method` to compressed-tensors, despite the later commit titles
primarily describing documentation/template changes. That newer format is not
silently substituted into this older runtime. The draft is pinned at its
current MXFP8 revision. Upstream did not pin its model snapshots, so these
choices narrow reproducibility risk without proving exact historical weights.
The reported ModelOpt corruption issue makes multilingual and tool quality a
required gate for this candidate; a newer checkpoint is a separate comparison,
not assumed to be a compatible bug fix.

Original additions: `prepare.py` renders/download-pins the local rank plan,
checks local configuration/shard presence and image identity when explicitly
executed, and gives this experiment separate container names, API/rendezvous
ports and compiler-cache paths. It never SSHes, downloads, stops services,
changes host settings, or changes the existing EXL3 launcher. The two rank
plans must use the same derivative image ID. Hashing config plus checking
shards is not an independent full-weight integrity attestation; retain the
revision-specific Hugging Face download metadata on both nodes.

## Local preparation

From the repository root:

```bash
python3 tests/test_nvfp4_candidate.py
python3 research/experiments/nvfp4-tp2/check_source.py --vllm-source /path/to/vllm-checkout
./scripts/check.sh all
```

The source probe reads the pinned base through Git into a temporary directory,
applies the existing SparkGLM DFlash2 installer then the copied MXFP8 installer,
and checks a second application changes nothing. It does not import torch or
execute model code. It requires the vLLM base commit already present locally.

## Later build and launch, only when hardware is available

Build the copied derivative on an ARM64 CUDA-capable build host:

```bash
docker build --platform linux/arm64 -t sparkglm:nvfp4-tp2 research/experiments/nvfp4-tp2
docker image inspect sparkglm:nvfp4-tp2 --format '{{.Id}}'
```

Use that full image ID for **both** plans. Example addresses below are
documentation placeholders; use each node's actual fabric interface/HCA.

```bash
python3 research/experiments/nvfp4-tp2/prepare.py \
  --rank 0 --image-id sha256:REPLACE_WITH_BUILT_IMAGE_ID \
  --head-ip 192.0.2.1 --worker-ip 192.0.2.2 --interface eth1 --hca mlx5_0 \
  --models /srv/sparkglm-nvfp4/models --cache /srv/sparkglm-nvfp4/cache
```

Without `--execute`, this prints JSON and performs no host/network operations.
The plan includes immutable `hf download` commands for the target and draft.
When authorized, stage those snapshots on both nodes and transfer the same
derivative image. Preserve local tokenizer/template files from each snapshot.
Run the worker's plan first with `--rank 1`, its own interface/HCA/cache paths,
and `--execute`; then run the head. Both Sparks must be free of resident model
workloads. This experiment does not stop or drain the existing appliance.
Poll `/health` on port 8889; retain startup/backend logs from both ranks.
Automatic restart is disabled so a failed candidate remains observable.

## Qualification before a switch

1. G1: inspect actual Humming selection and quantization; test relevant expert
   shapes/numeric tolerances and graph replay. Do not label Humming as native
   FP4 without compiled-instruction or profiler evidence.
2. G2: adapt/qualify tinyGLM for this quantization. The existing EXL3 tinyGLM
   receipts do not qualify NVFP4 or MXFP8 draft execution.
3. G3: reproduce the copied profile first, then compare appliances using the
   existing harness. `matrix.py` prints 16K/32K/64K C1, 16K/32K C2, 16K C4,
   and active-decode-plus-incoming-32K commands. Discard r0, alternate retained
   arms over at least three pairs, and retain exact prompts/token counts.
   Primary metric: existing stream's maximum/p95 visible SSE gap during incoming
   prefill, protecting new-request TTFT, delivery throughput and completion.
   Verify that the first stream is still decoding when the second arrives;
   otherwise reject the overlap sample. Use fresh per-run prefixes for cold
   tests, then separately test repeated-prefix conversations.
4. The copied upstream profile changes drafter precision/TP, batch size and
   other settings versus current EXL3. First results are **whole-stack** A/B.
   To attribute a win to weights/backend, subsequently hold BF16 DFlash2 K7,
   draft TP2, FP8 KV, context, scheduler, graph coverage and memory budgets
   fixed on both arms. Never call unlike recipes a quantization-only result.
5. G4: multilingual/U+FFFD, reasoning, executable code, parsed tool calls,
   request isolation, prefix hit/miss and cancellation tests. The copied
   text-only profile does not qualify vision; re-enable and qualify it before
   claiming feature parity. Compare target-logit fidelity with the same KV
   dtype and teacher if obtaining KLD.
6. Record exact image/config/model identities, memory peaks and both-rank logs
   in a checksum-bound qualification bundle. Promotion still requires the
   repository's G3/G4 review and G5 for a new recommended appliance baseline.

Example command generation (no endpoint calls):

```bash
python3 research/experiments/nvfp4-tp2/matrix.py \
  --base-url http://127.0.0.1:8889 --model sparkglm-nvfp4-candidate \
  --arm nvfp4 --output /tmp/nvfp4-results
```

## Provenance and licenses

Upstream source pin and per-file hashes are in `candidate.json`, with the
repository-wide entry in `provenance/upstreams.json`. Mia's MIT notice is
retained in `LICENSE`; the copied DFlash2 model retains its vLLM Apache header.
Original orchestration is Apache-2.0. The existing standalone benchmark remains
AGPL and is invoked as a separate program; no benchmark code was copied here.
Checkpoints remain separately downloaded publisher artifacts, not repository
contents. No performance or semantic qualification is claimed by local checks.
