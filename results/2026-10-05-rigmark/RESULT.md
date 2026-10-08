# 2026-10-05: RigMark decode and prefill on the carveout-placement release

[RigMark](https://github.com/alexellis/rigmark) (Alex Ellis / OpenFaaS, MIT), run from othexmr's fork at `40fabcaf`,
single-stream decode and cold prefill cells, the same command as
[2026-10-04](../2026-10-04-rigmark/RESULT.md) with comparison ID `atlas-prod-1005`.

- **Engine and image:** Atlas `sparkglm/atlas-20261005-cvplace` @ `72588541`,
  `ghcr.io/enntity/atlas-sparkglm:f047a0c5e796`, 4 × 512K profile at 0.91 with `SPARKGLM_DISPLAY_CARVEOUT=1`
  (the carveout holds latent KV pools only; 85,415 blocks per Spark), launched by LLooM.
- **KV pool and context ceiling** (not on the card): 524,288 tokens per request, up to 4 requests sharing one pool of
  85,415 blocks × 16 = 1,366,640 tokens per Spark, fp8_g128 MLA latent. That is about 9.13 GB of KV per rank
  (7.94 GB latent at 8,448 B per block per layer over 11 layers, about 1.2 GB sparse index), 2 GiB of it in the
  display carveout. See [bench/RIGMARK.md](../../bench/RIGMARK.md) for declaring these in `metadata.json`.
- **Receipt metadata is stale.** `run.appliance.serving_engine` still names the 2026-10-04 engine (`fdd2f965`,
  image 35824765d542); the metadata file was corrected after this run. The receipt is published unedited.
- **Hardware:** two DGX Sparks joined by one 200G cable. The endpoint was the production server; no other benchmark
  ran, but ordinary LLooM traffic was not blocked.

| Cell | This release | 2026-10-04 release |
|---|---|---|
| Decode, code (tok/s, median of 3; runs) | **65.7** (65.7 / 62.5 / 66.6) | 64.5 |
| Decode, prose | **36.3** (36.3 / 36.1 / 37.2) | 35.3 |
| Decode, structured | **91.6** (91.6 / 94.5 / 91.0) | 88.4 |
| Cold prefill 8K, time to first token (2 runs) | **3.28 s** (3.28 / 3.28) | 3.35 s |
| Cold prefill 32K | **12.35 s** (12.33 / 12.36) | 12.70 s |
| Cold prefill 64K | **23.71 s** (23.66 / 23.77) | 25.13 s |
| Replay of the same prompt, 8K / 32K / 64K | 0.26 / 0.28 / 0.28 s | 0.26 / 0.26 / 0.27 s |

Cold prefill is back: 64K is 5.6% faster than the 2026-10-04 release and 2.7% faster than 2026-09-30 (24.38 s). The
decode code path did not change in this release (greedy outputs and tokens per step are identical); the decode rows
moved within run-to-run spread.

## Staggered arrivals and the staggered C4 field guide

Same production instance, run right after (`raw/go_cap1005.sh`).

RigMark staggered arrivals (32K long request, 256-token short chats, 1 s delay, 2 rounds each, `raw/rig-atlas-1005-stag.*`):

| Level | Prefill first: newcomer first token (median) | Long request vs solo | Decode first: long request first token | vs solo |
|---|---:|---:|---:|---:|
| 2 | **2.86 s** | 1.08× | 12.65 s | 1.02× |
| 4 | **3.51 s** (max 4.21) | 1.18× | 12.76 s | 1.03× |

The 2026-09-30 release measured 2.88 / 3.95 s prefill-first and 1.02 / 1.03× decode-first. Level 6 was not run (the
4 × 512K profile serves four at a time).

Staggered C4 field guide (four ~16K requests at 0 / 1 / 2 / 3 s, 400 tokens each; discarded warmup 49.56 s;
`raw/field/`): 49.48 / 49.79 / 49.35 / 49.10 / 49.03 s, median **49.35 s**, 1,600 / 1,600 tokens every run.

## Verify the receipts

The JSON and card are RigMark's own output, unedited. In the logs and the capture script, the home directory was replaced with `/home/user` and the fabric address with `192.0.2.1`.

```sh
cd results/2026-10-05-rigmark && sha256sum -c SHA256SUMS
```
