# Published four-stream video configuration

A fresh checkout now defaults to the reconstructable runtime configuration used
for the corrected four-stream video published on 2026-09-03. The historical trace is
`results/legacy/2026-09-03-sparkglm-fixed-c4-video/` and identifies the original
recipe commit as `e7e35579b8058bbacb2408dce67b8fb7dd39f9b4`.

## Default runtime mapping

| Setting | Fresh-checkout default | Video configuration |
| --- | --- | --- |
| topology | `TP=2`, `NNODES=2` | TP2 on two GB10 systems |
| target quantization | `QUANTIZATION=exl3`, `KV_CACHE_DTYPE=fp8` | EXL3/TR3 K4, FP8 target KV |
| speculation | `SPEC_METHOD=dflash`, `DFLASH_TOKENS=7` | DFlash2 k=7 |
| drafter placement | `DFLASH_DRAFT_TP=2` | TP2 |
| model length | `MAX_MODEL_LEN=1000000` | 1,000,000 |
| request capacity | `MAX_NUM_SEQS=4` | four resident streams |
| prefill chunk | `MAX_NUM_BATCHED_TOKENS=7168` | 7,168 |
| memory budget | `GPU_MEM_UTIL=0.87`; no manual KV override | 0.87; automatic 19.87 GiB pool in the recorded run |
| scheduling | `GLM53_MIXED_PREFILL_CHUNK=0` | work-conserving mixed prefill/decode |
| sparse-indexer workspace | `GLM53_INDEXER_WORKSPACE=rightsize` | right-sized |
| TP worker spin | `GLM53_SPINWAIT_MS=16` | 16 ms |
| EXL3 fat path | M64 paired/fused E2 enabled | enabled |
| later candidate kernels | grouped prefill and cooperative decode disabled | not present in the historical recipe |
| vision tower | `LANGUAGE_MODEL_ONLY=0` | loaded, although the benchmark itself was text-only |
| long-shape startup gate | `GLM53_BOOT_LONG_C4=1` | four staggered 16K streams had to become resident |

The launcher rejects a manual KV-cache allocation below 18 GiB when four
sequence slots are requested. This prevents the undersized-cache mistake that
made the first SparkGLM capture queue two of the four streams.

## What “same as the video” does and does not mean

The default is the same **reconstructable runtime profile**. The current source tree is not a
byte-for-byte rebuild of the historical image: it includes subsequent fixes,
the pinned Z.AI chat template, security hardening, attribution work, and
candidate kernels that remain disabled. The container labels record the current
source revision and recipe hash so a new result cannot masquerade as the old
one.

The retained video trace is a single warmed visual run, not a post-policy
G3/G4/G5 qualification of current HEAD. It recorded four approximately 16K
actual-token prompts arriving at 0/1/2/3 seconds, 400 output tokens per stream,
88.22 seconds full wall time, and 23.58 aggregate delivered output tok/s after
the first visible token. Those figures describe that run only. Rebuilding the
same profile does not promise identical timing, model output, or speculative
acceptance on another checkout or appliance.

The historical receipt did not preserve the container digest, complete
environment manifest, or immutable target/drafter checkpoint revisions. The
runtime values above are recoverable from the recorded recipe commit and its
capacity-fix report; the current launcher additionally pins the public model
and DFlash2 revisions. We cannot retroactively prove that those exact weight
bytes were loaded for the video. A new full qualification of current HEAD is
still required before claiming byte-for-byte or performance equivalence.

## License boundary

The default fetches `incoai/GLM-5.3-Flash-DFlash2` at the pinned revision in
`.env.example`. Its checkpoint is CC BY-NC-ND 4.0, including non-commercial and
no-derivatives restrictions. SparkGLM does not redistribute those weights.
Operators whose use is incompatible with those terms must choose:

```bash
SPEC_METHOD=mtp ./start.sh
# or
SPEC_METHOD=none ./start.sh
```

Those overrides are supported, but they are not the configuration shown in the
published video and must not inherit its performance numbers.

Existing clones keep their local `.env`. After pulling this change, set
`SPEC_METHOD=dflash` and `LANGUAGE_MODEL_ONLY=0` there—or deliberately create a
fresh `.env` from `.env.example`—to select the published-video profile.
