# 2026-10-05: RigMark decode and prefill on the carveout-placement release

[RigMark](https://github.com/alexellis/rigmark) (Alex Ellis / OpenFaaS, MIT), run from othexmr's fork at `40fabcaf`,
single-stream decode and cold prefill cells, the same command as
[2026-10-04](../2026-10-04-rigmark/RESULT.md) with comparison ID `atlas-prod-1005`.

- **Engine and image:** Atlas `sparkglm/atlas-20261005-cvplace` @ `72588541`,
  `ghcr.io/enntity/atlas-sparkglm:f047a0c5e796`, 4 × 512K profile at 0.91 with `SPARKGLM_DISPLAY_CARVEOUT=1`
  (the carveout holds latent KV pools only; 85,415 blocks per Spark), launched by LLooM.
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

## Verify the receipts

The JSON and card are RigMark's own output, unedited. In the log, the home directory was replaced with `/home/user`.

```sh
cd results/2026-10-05-rigmark && sha256sum -c SHA256SUMS
```
