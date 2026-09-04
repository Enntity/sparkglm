# tinyGLM fast experiment fixture

tinyGLM is a deterministic, weightless integration model for testing the
GLM-5.3 serving stack without loading the 164 GiB checkpoint. It is deliberately
useless as a language model.

It preserves the dimensions and operations that choose the production kernels:

- hidden size 4096;
- EXL3 K4 trellis routed experts, width 2048 before TP2 sharding;
- sigmoid top-8 routing and the GLM SwiGLU clamp;
- one KDA layer and one sparse-NoPE-MLA layer;
- the production KDA, MLA/indexer, k-pool, mHC, TP2, and all-reduce geometry.

It reduces 45 layers to two, 288 experts to 16, the vocabulary to 256, removes
vision and speculative decoding, and uses deterministic dummy weights. Thus it
can validate code paths, shapes, collectives, CUDA graphs, memory safety, and
relative kernel changes. It cannot predict full-model weight-bandwidth pressure,
router/top-k cost over 288 experts, absolute endpoint throughput, DFlash2
acceptance, or output quality.

## Build

```bash
scripts/tinyglm.sh build
```

This writes a small Hugging Face-compatible metadata snapshot into the same
cache layout used by `start.sh`. It contains no safetensors. Generation should
complete in substantially less than a second.

## Serve on two Sparks

Serving replaces the containers named by the normal recipe. It does not delete
the full checkpoint, but restoring the full engine still pays its normal load
time.

```bash
scripts/tinyglm.sh restart
```

The launcher forces the safety-critical combination:

- `--load-format dummy`
- `SPARKGLM_TINY_DUMMY=1`
- TP2, EXL3, fp8 KV, and the current optimized kernel overlays
- no DFlash/MTP, no vision, and no long boot warmup

Run a quick staggered streaming check after `/health` becomes ready:

```bash
python3 benchmarks/tinyglm_smoke.py \
  --concurrency 4 --prompt-tokens 4096 --output-tokens 128 --stagger-ms 250
```

Use the same command before and after a candidate change. Its values are only
tinyGLM-relative A/B evidence; they are not production GLM tok/s predictions.

Never set `SPARKGLM_TINY_DUMMY=1` while loading a real checkpoint: it
intentionally overwrites routed-expert tensors with synthetic values.

For selector work, increase the routed-expert topology to the production count:

```bash
TINYGLM_EXPERTS=288 scripts/tinyglm.sh restart
```

That variant is less tiny because it allocates all 288 packed expert payloads.
Use the 16-expert default for kernel-integration iteration and a dedicated
288-way selector microbenchmark for online-top-k work.

Restore the real model with the normal recipe:

```bash
./start.sh restart
```

## Candidate promotion gate

The default development ladder is now:

1. Static tinyGLM/config validation before building an image.
2. A deterministic tinyGLM baseline on the current image.
3. Rebuild/restart tinyGLM with the candidate and compare three repeated cases.
4. Load the full checkpoint only after tinyGLM passes.
5. Promote only after the full warmed workload and quality gates pass.

```bash
# Before any GPU build
scripts/tinyglm-gate.sh static

# Current tinyGLM image
scripts/tinyglm-gate.sh record .tinyglm-gates/baseline.json

# Candidate tinyGLM image
scripts/tinyglm-gate.sh compare \
  .tinyglm-gates/baseline.json .tinyglm-gates/candidate.json
```

The TP2 gate runs three repeated shapes:

- C1 decode: 128-token prompt, 256 generated tokens;
- staggered C4 mixed work: 4096-token prompts and 128 generated tokens;
- staggered C2 long prefill: 16K-token prompts and 32 generated tokens.

It fails on changed generated token IDs, nondeterministic output, more than 5%
median throughput regression, or corresponding wall/TTFT regressions (with a
small absolute jitter allowance). Passing tinyGLM never replaces the final
full-model gate because depth, 288-way routing, memory bandwidth, DFlash2, and
real weight distributions are intentionally absent or reduced.
