# Posted four-stream video configuration

The build target is the implementation used for the final current-best
comparison posted on 2026-09-03, not merely a similarly configured successor.
The capture identifies the
optimization commit as
`4b2375977df873be57bc277b17778f94323ef7e6` plus its required launcher fix
`c80531867e13085c356ae5b9bff4c3b98ee64e8b`.

This is deliberately different from the earlier corrected-C4 replay at
`e7e35579b8058bbacb2408dce67b8fb7dd39f9b4`. That predecessor proved four
resident streams and produced the 88.22-second trace retained under
`results/legacy/2026-09-03-sparkglm-fixed-c4-video/`, but it is not the later
"current best" capture used for the posted comparison.

## Default runtime mapping

| Setting | Fresh-checkout default | Posted configuration |
| --- | --- | --- |
| topology | `TP=2`, `NNODES=2` | TP2 on two GB10 systems |
| target quantization | `QUANTIZATION=exl3`, `KV_CACHE_DTYPE=fp8` | EXL3/TR3 K4, FP8 target KV |
| speculation | `SPEC_METHOD=dflash`, `DFLASH_TOKENS=7` | DFlash2 k=7 at `7d74cdd881ed7e32c31175984a67823127b66cfe` |
| drafter placement | `DFLASH_DRAFT_TP=2` | TP2 |
| model length | `MAX_MODEL_LEN=1000000` | 1,000,000 |
| request capacity | `MAX_NUM_SEQS=4` | four resident streams |
| prefill chunk | `MAX_NUM_BATCHED_TOKENS=7168` | 7,168 |
| memory budget | `GPU_MEM_UTIL=0.87`; no manual KV override | 0.87; automatic pool |
| scheduling | `GLM53_MIXED_PREFILL_CHUNK=0` | work-conserving mixed prefill/decode |
| sparse-indexer workspace | `GLM53_INDEXER_WORKSPACE=rightsize` | right-sized |
| TP worker spin | `GLM53_SPINWAIT_MS=16` | 16 ms |
| EXL3 fat path | M64 paired/fused E2 enabled | enabled |
| grouped prefill | `EXL3_GROUPED_PREFILL_K4=1` | enabled |
| cooperative decode | `EXL3_DECODE_COOP_K4=1`, max tokens 16 | enabled on both ranks |
| vision tower | `LANGUAGE_MODEL_ONLY=0` | loaded, although the benchmark was text-only |
| long-shape startup gate | `GLM53_BOOT_LONG_C4=1` | four staggered 16K streams must become resident |

The launcher rejects a manual KV-cache allocation below 18 GiB when four
sequence slots are requested. This prevents the undersized-cache mistake that
made the first SparkGLM capture queue two of the four streams.

## Retained capture identity

The full capture receipt records:

- label `SparkGLM current best - mixed=0 - grouped prefill + coop decode`;
- engine identity `4b23759+c805318`;
- four approximately 16K actual-token prompts arriving at 0/1/2/3 seconds;
- 400 output tokens per stream, 86.148650 seconds wall time, and 24.267091
  aggregate delivered output tokens/s;
- a cache-unique salt, so the run did not reuse the competing arm's prompt
  prefix.

The clicks-oriented top/bottom file preserved locally as
`glm53-flash-tp2-four-stream-comparison.mp4` is a trim of that captured timing,
not a synthetic acceleration. Its SHA-256 is
`3dca189abab2e9350812116b5c7acf73cbe832ae9bc18dfcd44a7f507ce3583c`.

The retained two-Spark appliance and Docker journal establish the serving
identity:

- image `sparkglm-vllm:exl3-grouped-prefill-candidate-20260903d`, ID
  `sha256:0b17bd9246763d74e2f5e1b79fecdcb6a8ef03e1b8e5823f2d2183ceafb91159`;
- image source label
  `6a49e49e7e6a3226197a2ceefcf217cdf55f751e`;
- live container environment values `GLM53_MIXED_PREFILL_CHUNK=0`,
  `EXL3_GROUPED_PREFILL_K4=1`, `EXL3_DECODE_COOP_K4=1`, and
  `EXL3_DECODE_COOP_MAX_TOKENS=16`;
- target revision
  `25a44fdbf16862a46b7cc9921142c6c81350af2f`;
- DFlash2 revision
  `7d74cdd881ed7e32c31175984a67823127b66cfe`, with model SHA-256
  `8931dc522be0aa31760a7463f8d2f8044fa3e6d40be2e87aa08e9fd17bfd6683`.

All 120 target shards and every serving/tokenizer file were independently
hash-verified against the retained manifest on both ranks. The drafter file was
also identical on both ranks.

## What “same as the video” does and does not mean

**Correction:** the earlier clean reconstruction at `b3d0bbd` did not contain
all of the video implementation. It restored the EXL3 overlays and runtime
settings on Mia's base, but omitted changes inherited from the video's custom
vLLM foundation. Similar throughput did not establish source parity. That
reconstruction did not meet the requested default-build contract.

The restoration in `patches/video/` includes the inherited FP16 sparse-indexer
path, its native top-k and DeepGEMM support, scheduler alignment/phase support,
per-group prefix-cache retention, and FlashAttention discovery. The Dockerfile
rebuilds the changed native components from pinned public sources. It checks
the installed Python inventory against all 2,423 files in the retained video
image, allowing only three explicit byte-pinned metadata/comment/license
differences. It also checks the eight changed native-source inputs against
the historical foundation. See `provenance/video-source-parity.json` and
`docs/VIDEO_SOURCE_RESTORATION.md` for the exact boundary and validation status.

Source parity is not binary reproducibility: the build retains current source
labels and subsequent chat-template, security, build, attribution, and
qualification work. It must not impersonate the historical image.

The video is one warmed visual sample, not a post-policy G3/G4/G5
qualification. Rebuilding the same profile does not promise identical timing,
model output, speculative acceptance, clocks, or thermals. A fresh full-model
replay of current HEAD is required before claiming current performance
equivalence. The grouped-prefill and cooperative-decode limitations are stated
in `docs/KNOWN_LIMITATIONS.md` and both retain explicit rollback switches.

A replay of the **incomplete reconstruction** is retained under
`results/candidates/2026-09-04-clean-release-c4-replay/`. Three warmed runs had
a 23.514 aggregate decode tok/s median versus 24.267 tok/s in the posted
one-off, a 3.1% difference. Those measurements remain valid for that build, but
are not evidence that it reproduced the video implementation. Do not attach
them to the restored build or treat them as a statistically certified tie.

The [source-restored replay](../results/candidates/2026-09-04-video-source-restoration/RESULT.md)
is now separate evidence: three warmed runs took 90.678 seconds median versus
87.552 seconds for a contemporary replay of the retained original container.
The corresponding aggregate rates are 22.770 and 24.023 tok/s. The rebuild
restores the audited source; these initial runs did not isolate the slower
point estimate or certify speed or quality equivalence.

The stronger [ten-repetition-per-image comparison](../results/candidates/2026-09-04-video-runtime-isolation/TEN_REP_RESULT.md)
finds mean completion times of 88.835 seconds preserved and 88.907 seconds
rebuilt (+0.08%) with equal actual cache pools and CPU placement. The earlier
5% throughput deficit did not reproduce. This is a headline tie for the
controlled video workload, not an equivalence certificate for every latency
metric, automatic memory-sizing condition, or model-quality case.

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
posted video and must not inherit its performance numbers.

Existing clones keep their local `.env`. After pulling this change, explicitly
set `EXL3_GROUPED_PREFILL_K4=1`, `EXL3_DECODE_COOP_K4=1`, `SPEC_METHOD=dflash`,
and `LANGUAGE_MODEL_ONLY=0` there—or deliberately create a fresh `.env` from
`.env.example`—to select the posted-video profile.
