# 2026-10-04: RigMark decode and prefill on the current release

[RigMark](https://github.com/alexellis/rigmark) (Alex Ellis / OpenFaaS, MIT), run from othexmr's fork at `40fabcaf`,
single-stream decode and cold prefill cells only:

```sh
./rigmark run --base-url http://127.0.0.1:8893 --model auto --metadata metadata.json \
  --comparison-id atlas-prod-1004 --runs 3 --prefill-runs 2 --prefill-depths 8192,32768,65536 \
  --skip-concurrency --extra-body '{"chat_template_kwargs":{"reasoning_effort":"low"}}' \
  --label atlas-1004-decode --output rig-atlas-1004-decode.json
```

- **Engine and image:** Atlas `sparkglm/atlas-20261004-gram` @ `fdd2f965`,
  `ghcr.io/enntity/atlas-sparkglm:35824765d542`, 4 × 512K profile at `GPU_MEMORY_UTILIZATION=0.91` with
  `SPARKGLM_DISPLAY_CARVEOUT=1`, launched by LLooM.
- **Receipt metadata is stale.** The receipt's `run.appliance.serving_engine` and `serving_image` fields
  were copied from the 2026-09-30 run's metadata file (engine `sparkglm/atlas-20260930c`, 0.93). This run
  used the image above. The receipt is published unedited.
- **Hardware:** two DGX Sparks joined by one 200G cable. The endpoint was the production server; no other
  benchmark ran, but ordinary LLooM traffic was not blocked.

## Results

| Cell | This release | 2026-09-30 release ([receipt](../2026-09-30-rigmark/RESULT.md)) |
|---|---|---|
| Decode, code (tok/s, median of 3; runs) | **64.5** (66.1 / 63.6 / 64.5) | 62.9 |
| Decode, prose | **35.3** (35.3 / 34.9 / 36.0) | 34.0 |
| Decode, structured | **88.4** (89.5 / 88.4 / 87.6) | 87.5 |
| Cold prefill 8K, time to first token (2 runs) | **3.35 s** (3.40 / 3.30) | 3.38 s |
| Cold prefill 32K | **12.70 s** (12.71 / 12.69) | 12.71 s |
| Cold prefill 64K | 25.13 s (25.15 / 25.11) | **24.38 s** |
| Replay of the same prompt, 8K / 32K / 64K | 0.26 / 0.26 / 0.27 s | 0.27 / 0.28 / 0.28 s |

Decode is 1.0–3.8% faster, from the step-cut and gap-close engine options. Prefill at 8K and 32K is
unchanged. **Prefill at 64K is 3.1% slower**: both runs took 25.1 s, against 24.38 s on 2026-09-30. The
cause is not yet known. Candidates are the display-carveout KV placement, the lower memory utilization
(0.91 against 0.93), and changes in the engine series between the two releases.

Concurrency, short-code and staggered-arrival cells were not re-run; their latest numbers are in the
2026-09-30 receipt.

## Verify the receipts

The JSON and card are RigMark's own output, unedited. In the log, the home directory was replaced with
`/home/user`.

```sh
cd results/2026-10-04-rigmark && sha256sum -c SHA256SUMS
```
