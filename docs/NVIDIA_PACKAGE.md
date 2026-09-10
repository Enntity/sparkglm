# NVIDIA NVFP4 + DFlash2 package

Selected for the maintainer's NVIDIA-weight configuration on September 9.
This source package reuses the tested native FP4 serving path and does not
include the experimental MTP loader or numerical diagnostic patches.
NVIDIA is a checkpoint preference, not a demonstrated quality improvement.
The public main branch's historical Red Hat default is unchanged by this branch.

## Run

Use `./start-nvidia.sh` wherever the [two-Spark guide](../SPARKGLM.md) uses
`./start.sh`. The wrapper always selects `nvfp4-nvidia`, including lifecycle
commands. On a Spark leader:

```bash
./start-nvidia.sh plan
./start-nvidia.sh install --worker USER@WORKER \
  --head-address HEAD_FABRIC_IP --worker-address WORKER_FABRIC_IP \
  --interface HEAD_INTERFACE --worker-interface WORKER_INTERFACE \
  --hca HEAD_RDMA_DEVICE --worker-hca WORKER_RDMA_DEVICE \
  --gid HEAD_GID_INDEX --worker-gid WORKER_GID_INDEX
./start-nvidia.sh status
./start-nvidia.sh stop
./start-nvidia.sh start
```

Reuse already downloaded weights with `--model-root /path/to/models` and a
qualified local image with `--image sha256:FULL_IMAGE_ID`. The LLooM alternative
is `./start-nvidia.sh --lloom` with its documented installation arguments.
Do not install a second lifecycle owner over an active model.
The served model ID is `sparkglm-nvfp4-nvidia`; the profile specifies port 8892.

## Pinned configuration

| Component | Selected value |
| --- | --- |
| Target | `nvidia/GLM-5.3-Flash-NVFP4` |
| Target revision | `423acf37583782c51c142d145aef733d72943d93` |
| Quantization and MoE | `modelopt_fp4`, native `flashinfer_cutlass` |
| Draft | `local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8` |
| Draft revision | `610aa967a92bfeb97e3d848dcb8693553e8b6a55` |
| Target / draft parallelism | TP2 / TP2 on two GB10 Sparks |
| Speculative tokens | 7 |
| KV | FP8, 9 GiB per rank |
| Configured maximum context | 524288 tokens; NVIDIA full-window capacity unqualified |
| Concurrent sequences / prefill chunk | 4 / 2048 tokens |
| Scheduling | Mixed prefill/decode, CUDA graphs and prefix cache enabled |

Machine-readable recipe: [nvfp4-nvidia.json](../profiles/nvfp4-nvidia.json).
Immutable source build layers: [build.json](../profiles/build.json).
Rebuilt images receive their own identities; existing measurement manifests
identify the exact tested image pair. No weight files are redistributed.
The DFlash2 draft retains its non-commercial license restrictions;
see [licensing](LICENSING.md).

## Actual checkpoint comparison

Same image pair, workload prompt hashes, DFlash2, cache budget, context and
scheduler settings. One retained run per arm, Red Hat then NVIDIA, after a
C4 warmup. These are bounded observations, not a statistical speed ranking.
The percentage is NVIDIA's change in complete wall time; negative is faster.

| Workload | Red Hat seconds | NVIDIA seconds | Wall change |
| --- | ---: | ---: | ---: |
| C1 16K | 20.409 | 22.792 | +11.7% |
| C1 32K | 26.585 | 29.123 | +9.5% |
| C2 16K | 30.877 | 33.521 | +8.6% |
| C2 32K, primary | 55.191 | 50.739 | -8.1% |
| C4 16K | 61.212 | 62.534 | +2.2% |

[Full precision CSV, including TTFT](nvidia-comparison.csv) is derived from
[checksum-bound raw receipts](../results/candidates/2026-09-09-nvidia-modelopt/qualification.json).
All requests completed their 400 output tokens without foreign request markers.
Bounded semantics: NVIDIA 14/16, Red Hat 15/16. Both missed one arithmetic
question; NVIDIA missed a second. This is not proof of general quality
superiority or equivalence. Existing integration passed G1/G2; G3 repetition,
G4 quality and G5 release qualification remain incomplete. See the
[original report](../results/candidates/2026-09-09-nvidia-modelopt/RESULT.md).

## Comparison video

The new comparison pairs NVIDIA with Red Hat on SparkGLM. It uses the traditional
four field-guide streams, approximately 16K input each, 400 output each,
arriving at 0/1/2/3 seconds. It is a different workload from the isolation table
above. Both arms use identical prompt text and corresponding cache salts;
actual server token counts are retained. Each checkpoint gets one discarded
warmup and three retained captures; display the median wall-time capture from
each at 1x elapsed time, with all six results available beside the video.
Run order is blocked by checkpoint to avoid repeated full-model loads; do not
claim alternating-pair G3 qualification. Never relabel an older Red Hat/Mia
video as NVIDIA footage. Capture results and media are recorded separately.

### Completed video comparison

The fresh three-run medians were NVIDIA **90.500 seconds** and Red Hat
**89.243 seconds**, a **1.41% increase in NVIDIA wall time**. Individual runs
vary more than this gap. All 24 retained requests completed. The new semantic
panel again scored NVIDIA 14/16 versus Red Hat 15/16. This supports the selected
package under the maintainer's bounded-performance tolerance; it does not
establish a quality gain or formal performance equivalence.

[All runs, capture method and video receipt](../results/candidates/2026-09-09-nvidia-redhat-video/RESULT.md).
