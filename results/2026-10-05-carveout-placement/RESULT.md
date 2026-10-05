# 2026-10-05: display carveout holds latent KV pools only (engine sparkglm/atlas-20261005-cvplace)

**Problem** ([Enntity/sparkglm#30](https://github.com/Enntity/sparkglm/issues/30)). RigMark's 64K cold prefill took
25.13 s on the 2026-10-04 release against 24.38 s on 2026-09-30. On the idle pair the loss was entirely the display
carveout (`DISPLAY_CARVEOUT=1`), and it grew with context: 3% at 32K, 7% at 64K.

**Cause.** The GPU maps the carveout L2-uncached. A microbenchmark on spark03 (`raw/cvbench.cu`,
`raw/cvbench-spark03.txt`) imports it exactly as the engine does and runs the same tests on an equal `cudaMalloc`
buffer:

| Test | cudaMalloc | Carveout |
|---|---:|---:|
| Streaming read, 1.5 GiB | 248 GB/s | 257 GB/s |
| Re-read a 4 MiB region 256 times | 1,679 GB/s | 266 GB/s |
| Dependent load, 16 MiB / 1.5 GiB span | 159 / 161 ns | 408 / 411 ns |

One pass streams at full speed; data read again comes from DRAM every time. In open-gpu-kernel-modules 580.173.02,
memory-list descriptors start `NV_MEMORY_UNCACHED` and the sysmem list path never changes it. The old placement put
the largest buffers in the carveout: two latent pools and five sparse-index layers. The indexer re-reads every earlier
key for every query, so the index layers carried most of the loss.

**Change.** The carveout now takes latent (`K`/`V`) pools only, largest first (`ATLAS_KV_CARVEOUT_ORDER`, default
`latent`; `size` restores the old order for comparison). Placement moves allocations only; outputs are unchanged.

## Measurements

Two DGX Sparks, 4 × 512K profile at 0.91, started by `./start.sh`, RigMark cold prefill, 3 runs each (`raw/pf64.log`,
`raw/cvplace.log`; per-arm receipts `raw/*.json`).

| Arm | KV blocks per Spark | 32K | 64K |
|---|---:|---:|---:|
| 2026-10-04 release (image 35824765d542), carveout on | 91,066 | 12.70 s | 25.11 s |
| same image, carveout off | 70,345 | 12.32 s | 23.45 s |
| 2026-09-30 release (image 43eed6283b96) at 0.93 | 100,456 | 12.68 s | 24.11 s |
| this engine, index buffers first | 89,306 | 13.03 s | 26.32 s |
| this engine, index buffers only | 82,762 | 13.04 s | 26.27 s |
| this engine, carveout off | 70,501 | 12.27 s | 23.30 s |
| **this engine, latent pools only (default)** | **86,005 (+19.4%)** | **12.25 s** | **23.45 s** |

With the carveout on, the pool is about 1.38M tokens instead of 1.46M, and cold prefill is within 0.7% of the
carveout off and 2.7% faster at 64K than the 2026-09-30 release. The engine arms ran a test image that layers the
fix branch's binary (`57d1fe4d`) onto the 2026-10-04 image; the pinned commit `72588541` adds comments only.

The two index arms disproved the first hypothesis (that scattered gathers from the latent pools were slow) and are
kept as receipts.

## Not done here

The driver's map call can override the uncached default (`NVOS46_FLAGS_GPU_CACHEABLE_YES`), and UVM's external
mapping can force caching, but CUDA's `cuMemMap` passes neither. A user-space route that would let the index buffers
use the carveout too is tracked in [#32](https://github.com/Enntity/sparkglm/issues/32).

## Verify the receipts

The JSON files are RigMark's own receipts, unedited. In the logs and scripts, home directories, host names and
private addresses were replaced with `/home/user`, `spark` and `192.0.2.1`.

```sh
cd results/2026-10-05-carveout-placement && sha256sum -c SHA256SUMS
```
