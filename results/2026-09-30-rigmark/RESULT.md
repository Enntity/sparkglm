# 2026-09-30: RigMark on the decode-step release

[RigMark](https://github.com/alexellis/rigmark) (Alex Ellis / OpenFaaS, MIT),
run from othexmr's fork at `40fabcaf` with its staggered-arrival suite, the
same client and cell set RiNGSiDE's TP2 row was measured with
(`raw/run-atlas.sh`: code-full, then prose concurrency with staggered
arrivals; 2 runs per cell; `reasoning_effort: low`, thinking on).

- **Engine and image:** Atlas `sparkglm/atlas-20260930c` @ `9b8160e3`,
  `ghcr.io/enntity/atlas-sparkglm:43eed6283b96`, 4 × 512K profile at
  `GPU_MEMORY_UTILIZATION=0.93`, launched by LLooM. The receipts' `serving_image`
  field ("dev container with bind-mounted spark binary") is left over from an
  earlier run's metadata file; this run used the published image unchanged.
- **Hardware:** two DGX Sparks joined by one 200G cable. No other traffic.

## Results, and RiNGSiDE's published TP2 row on the same client

| Cell | SparkGLM (Atlas) | RiNGSiDE vLLM TP2 (published) |
|---|---|---|
| Decode, code / prose / structured (tok/s, median of 2) | **62.9 / 34.0 / 87.5** | 56.5 / 33.0 / 83.4 |
| Cold prefill 8K / 32K / 64K, time to first token (s) | **3.38 / 12.71 / 24.38** | 3.56 / 12.94 / 25.66 |
| Replay of the same prompt 8K / 32K / 64K (s) | 0.27 / 0.28 / 0.28 (prefix cache) | |
| Short code, 256 tokens per agent, aggregate at 1 / 2 / 3 / 4 streams (tok/s) | 45.8 / 56.5 / 61.9 / 67.0 | 44.0 at 1 stream |
| Short code at 5 / 6 streams (tok/s) | 61.2 / 62.0 | 97.3 at 6 streams |
| Short prose aggregate at 1 / 2 / 3 / 4 / 5 / 6 streams (tok/s) | 32.0 / 43.3 / 47.8 / 52.1 / 47.9 / 49.8 | |
| Staggered, prefill first: newcomer time to first token at 2 / 4 / 6 (s) | **2.88 / 3.95 / 4.36** | 4.70 / 5.15 / 4.90 |
| Staggered, decode first: newcomer time to first token at 2 / 4 (s) | **12.97 / 13.04** | 14.38 / 14.81 |

RiNGSiDE's figures are its published TP2 row (othexmr, measured 2026-09-25),
not a run on our pair.

## Limitations

- **Five and six streams.** The 4 × 512K profile serves four sequences at a
  time, so the fifth and sixth wait for a slot. The 5- and 6-stream cells
  measure that queue, not the engine's six-stream throughput, and the
  decode-first cell at 6 has no valid round (0 of 2). The 8 × 128K profile
  serves eight at a time and was not run.
- **Two runs per cell**, one server start.

## Verify the receipts

The JSON files are RigMark's own receipts, unedited (each card names its
SHA-256). The logs had a home directory replaced with `/home/user`.

```sh
cd results/2026-09-30-rigmark && sha256sum -c SHA256SUMS
```
