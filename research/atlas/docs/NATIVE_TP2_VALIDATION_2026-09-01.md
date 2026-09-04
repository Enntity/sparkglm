# GLM-5.3-Flash native two-Spark TP2 validation — 2026-09-01

## Result

The repository now contains a narrow, working GLM appliance for the two owned
DGX Sparks:

- pure tensor parallelism (`EP=1`, `TP=2`) over the CX7 link;
- `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` target weights;
- `incoai/GLM-5.3-Flash-DFlash2` BF16 drafter;
- seven drafts plus one predecessor row (`K=8`) by default;
- four live text sequences, 32K context, phase-interleaved scheduling;
- native SM121 KDA, sparse DSA, mHC, EXL3, target-verify, and DFlash graphs.

The final deployed binary is
`f8315919e14a38f127f3428f6301efee646900406c5ed6ce037cbdff8783a36f`.
Both ranks reported that exact SHA and rank 0 was healthy after the final
canaries.

This is substantially better than the first working Atlas path, but it is not
yet the fastest published two-Spark implementation. Mia's current vLLM overlay
still leads Atlas on the matched five-by-400 structured decode protocol. The
remaining gap is explicitly recorded below instead of being hidden behind a
short-output benchmark.

## What changed

### Prefill and target execution

- KDA projection work was merged from nine GEMMs to four and uses rank-local
  TP geometry.
- mHC finish plus RMS normalization is fused; BF16 weights widen to FP32 for
  the TF32 path.
- DSA prefill uses an eight-query causal tile for the first 2,048 visible
  tokens and a tensor-core sparse suffix. The legacy kernel remains available.
- The activation arena is 7,168 rows. Idle prefill can expose large routed
  experts to the device-compacted EXL3 fat path instead of being permanently
  limited to tiny chunks.
- The scheduler admits 7,164 prompt rows beside four reserved decode rows and
  interleaves active decode with 128-token prompt slices.

### Decode and concurrency

- DFlash target verification has fixed-address recurrent, sparse-index, and
  rollback metadata suitable for CUDA graph replay.
- Equal-width requests are packed sequence-major into one target pass. KDA
  causality and DSA/KV/index state remain per sequence.
- The DFlash proposer has native N=2/3/4 batch graphs and runtime-sized row
  graphs. It no longer computes the configured maximum row count when the
  scheduler requests a smaller concurrent block.
- Concurrent bootstrap is batched, so new requests can join one proposer and
  target weight sweep rather than serializing one full pass per request.
- The scheduler supports an experimental 16-draft ceiling with per-request
  demotion, but production defaults to seven drafts. The matched long-output
  gate proved that 16 drafts did not repay their larger target pass.

### Operator receipts

The real-checkpoint EXL3 probe measured the device-compacted fat path against
the persistent path:

| Routed rows | Fat path | Persistent path | Speedup |
|---:|---:|---:|---:|
| 129 | 0.508 ms | 2.032 ms | 4.0x |
| 256 | 0.680 ms | 3.538 ms | 5.2x |
| 512 | 1.193 ms | 7.114 ms | 6.0x |

This is a large-M prefill optimization. It is not the decode answer: an
eight-row verify rarely creates a fat expert.

On the deterministic 3,903-token isolation prompt, the original native path
had 13.728 s TTFT. The final fixed-`K=8` path measured 10.459 s TTFT and
14.99 decode tok/s after the structured concurrency warm-up. That is a 23.8%
TTFT reduction. The output contained its own isolation marker and no foreign
marker.

## Endpoint receipts

All rows below used temperature zero, thinking off, forced output length, and
the OpenAI streaming endpoint. Output isolation and rank logs are gates, not
optional observations.

### Matched Mia structured protocol

Prompt: `Count from 1 to 200. Output only the numbers, separated by spaces. No
other text.` Five warm runs generated 400 tokens each.

| Atlas configuration | Median C1 decode | TTFT median | Output |
|---|---:|---:|---|
| Experimental 16 drafts | 44.24 tok/s | 0.796 s | identical across runs |
| Final seven drafts | 49.60 tok/s | 0.670 s | identical across runs |

Seven drafts improved the honest long-output median by 12.1%. All five final
outputs had SHA-256
`0436adfad85df21cc0d591e5d2d3ab0092539d2a0e3de326b3944c66bfa6193a`.
The measured range was 48.21-50.81 tok/s.

One final post-deployment sweep kept the same prompt and 400-token response at
each concurrency. These are single-run concurrency receipts, not medians:

| Width | Aggregate decode | Per-request p50 | TTFT p95 | C1 scaling |
|---:|---:|---:|---:|---:|
| C1 | 49.51 tok/s | 49.51 tok/s | 1.471 s | 1.00x |
| C2 | 86.66 tok/s | 43.37 tok/s | 1.049 s | 1.75x |
| C4 | 121.34 tok/s | 31.38 tok/s | 1.989 s | 2.45x |

All seven responses completed 400 tokens without an error and had the same
SHA-256 as the five-run C1 gate. C4 therefore provides real aggregate
concurrency, but not linear scaling: the four requests share the same two GPU
weight sweeps and each request decodes more slowly.

After the final source-organization build, a fresh C1 canary measured 49.33
tok/s and reproduced the same output SHA-256. Both ranks ran binary
`f8315919…a36f`; neither rank logged a CUDA, NCCL, fallback, panic, or
collective-order error.

Mia's current repository reports 65.1 tok/s for its median-of-five 400-token
structured lab protocol. Atlas therefore reaches about 76% of that published
number, not parity. The prompt, temperature, and output length match; the
harness and software stack differ, so a same-pair A/B remains the final
cross-engine comparison.

### Short concurrency canary

The 64-token structured canary is useful for scheduling smoke tests, but it
must not be quoted as steady-state C1 decode because the TTFT boundary credits
the first speculative burst outside the measured decode window.

| Width | Aggregate decode | TTFT p95 | Output |
|---:|---:|---:|---|
| C1 | 68.94 tok/s | 0.873 s | completed |
| C2 | 111.04 tok/s | 0.870 s | byte-identical |
| C4 | 167.88 tok/s | 1.598 s | byte-identical |

At C4 every response had SHA-256
`755f01fcf98215c65f06fdf731665f1748752f8b7499efe99dd4ed40ce56c3d3`.
The structured fixture does not contain per-request markers, so it cannot
establish request-state isolation.

A separate final C4 isolation canary did: every response contained its own
request-specific marker, none contained another request's marker, and all four
requests completed without an error. TTFT p95 was 3.037 s for that 64-token
prompt and 64-96-token response test.

### Content-adaptive proof

With the experimental 16-draft ceiling, a long prose/isolation request
accepted `1, 2, 6` of the first three 16-draft blocks. The scheduler logged a
demotion and the very next verify used seven drafts. A specialized N=1/K=8
proposer graph was captured. Warm decode improved from 8.21 tok/s before the
fix to 14.22 tok/s, while the response retained its own marker and no foreign
marker. Production now starts directly at the proven seven-draft geometry.

## What Mia's 2026-09-01 update changes

The current Mia repository adds an E2 fat-expert EXL3 prefill path, makes it the
default, raises the default maximum batched tokens to 7,168, right-sizes sparse
index workspace, enables prefix caching, and retains DFlash2 k=7. PR 77 reports
a controlled roughly 18-19% fully uncached gain at 100K-300K for its retained
2,048-token setting on one deployment, with explicit warnings that another
deployment did not reproduce every gain.

Atlas independently has the corresponding large-M EXL3 path and 7,168-row
arena. The important part Atlas still lacks is Mia's mature target stack:

- FlashInfer's SM120 sparse MLA over packed FP8 target KV;
- its production fused EXL3 MoE implementation and tuning;
- validated prefix caching across GLM's hybrid state.

The first two are the likely source of the remaining decode-step gap. Prefix
caching improves repeated-prefix TTFT, not uncached decode throughput.

## Reproduce

Start the final appliance:

```bash
bash scripts/start-glm53-ep2.sh
```

Run the honest C1 comparison:

```bash
python3 bench/glm53/staggered_openai.py \
  --base-url http://127.0.0.1:8888 \
  --model GLM-5.3-Flash-EXL3 \
  --concurrency 1 --stagger-ms 0 \
  --prompt-style mia-structured \
  --prompt-tokens 64 --output-tokens 400 --min-output-tokens 400 \
  --disable-loop-watchdog
```

Warm once, then record five runs. Reject any run with an error, different
output hash, CUDA/NCCL error, proposer fallback, or target/draft collective
order mismatch.

Run C2 and C4 with the same command by changing only `--concurrency`. For
request-state isolation, use the default `isolation` prompt style; every request
must contain its own marker and no foreign marker.

## Rollback switches

- `GLM53_DFLASH_GAMMA=17`: experimental deep ceiling; default is 8 rows/7 drafts.
- `GLM53_DFLASH_DEPTH_LADDER=0`: disable width/content depth policy.
- `GLM53_DSA_PREFILL_TC=0`: disable the tensor-core DSA suffix.
- `GLM53_DSA_PREFILL_FAST=0`: restore the original DSA prefill kernel.
- `GLM53_EXL3_FAT=0`: disable the large-M EXL3 path.
- `GLM53_NO_DFLASH_MULTI_GRAPH=1`: eager concurrent DFlash target path.
- `GLM53_NO_MTP_BATCH_VERIFY=1`: serialize target verification.

## Remaining proof and performance gates

1. Profile final K=8 target verify by layer and close the roughly 24% gap to
   Mia's published five-by-400 result. Prioritize EXL3 MoE and sparse MLA.
2. Run the current Mia image and Atlas back-to-back on these exact two Sparks,
   same clocks, same prompt bytes, five warm runs, and retained raw receipts.
3. Add authoritative reference logits/greedy-token parity, mixed-length load,
   cancellation soak, endurance, and rank/node failure recovery.
4. Implement atomic KV + KDA + DSA/index prefix snapshots before enabling
   prefix reuse. A KV-only hit is incorrect for this architecture.
5. Keep vision and arbitrary topologies outside the fast appliance until the
   text path clears the preceding gates.
