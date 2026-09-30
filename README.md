# SparkGLM

GLM-5.3-Flash on two NVIDIA DGX Sparks, served by the
[Atlas](https://github.com/Enntity/atlas) inference engine: tensor and expert
parallelism across both boxes over one ConnectX-7 cable, NVIDIA's NVFP4
checkpoint, DFlash2 speculative decoding, prefix caching, up to four
concurrent requests with 512K context each, and an OpenAI-compatible API.
Multi-turn agents get a cached conversation back: a turn of a 45K-token
conversation starts in under a second instead of re-reading the whole
transcript.

```sh
git clone https://github.com/Enntity/sparkglm.git
cd sparkglm
cp .env.example .env      # set WORKER to the other Spark's ssh destination
./start.sh
```

## What it does

Measured on two DGX Sparks joined by one 200G cable, with images built from
a fresh clone. Receipts and caveats:
[a faster verify step and cheaper warm turns](results/2026-09-30-decode-step/RESULT.md) (this release),
[exact kernels and the prefix-cache policy](results/2026-09-30-exact-speedups/RESULT.md),
[index tails and 0.92 memory](results/2026-09-29-index-tails-092/RESULT.md),
[prefix caching](results/2026-09-29-prefix-caching/RESULT.md)
and [the first Atlas release](results/2026-09-29-atlas-merged/RESULT.md)
(decode, prefill, RigMark and quality rows).

| Workload | SparkGLM (Atlas) | Reference, same pair |
|---|---|---|
| Multi-turn conversation, 30–48K tokens: time to first token on turns 2+ (median) | **0.67 s** | 17.6 s with caching off |
| Four concurrent ~204K-token sessions, cached follow-up turns | 12/12 exact answers | |
| Replaying a 35K-token prompt | **0.80 s** (14.4 s cold) | |
| New session sharing a 24K system prompt with earlier ones: time to first token | **15.7 s** | 36.2 s with the cache policy off |
| Idle session resuming after another session's 17 turns (median) | **2.5 s** | 49.6 s with the cache policy off |
| Cold 61K / 125K prompt: time to first token | **25.0 / 49.1 s** | 29.2 / 62.5 s two engines ago |
| Matrix: C1/C2 at 16K and 32K, C4 at 16K, 400 tokens each, cold (sum of walls) | **146.0 s** | vLLM SparkGLM 206.6 s |
| Staggered C4: four ~16K requests arriving 1 s apart, cold | **52.7 s** (27.3–27.8 s warm) | vLLM SparkGLM 70.5 s · Mia EXL3 113.2 s |
| Single-stream prose decode, 5 × 384 tokens | **40.2–44.2 tok/s** | 32.8–34.5 tok/s on the previous release |
| Cold prefill at 8K / 32K / 64K / 139K | 2,500 / 2,470 / 2,310 / 2,020 tok/s | |
| Aggregate short-code decode, 1 / 6 / 8 streams (8 × 128K profile) | 36.9 / 76.7 / 88.0 tok/s | RiNGSiDE vLLM TP2 (published) 44.0 / 97.3 / – |
| Quality probe: arithmetic / two-hop 24K needle | 40/40 · 11/12 | |

These are measurements on our pair, not a guarantee for yours. Known gaps and
open questions are in [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

## Requirements

- Two DGX Spark (GB10, 128 GB) systems joined by a direct ConnectX-7 cable,
  with an IPv4 address on the cable's interface on each (RoCE; the engine
  finds the GID itself and uses both PCIe halves of the port).
- Docker with the NVIDIA runtime on both, and passwordless `ssh` from the
  Spark you run `./start.sh` on to the other one, whose user can run `docker`.
- About 230 GB free on each Spark: the checkpoint takes 204 GB, the drafter
  2 GB and the converted overlay 16 GB. The 207 GB download is made once and
  copied to the other Spark over the cable (about a minute).
- Nothing else using the GPUs' memory. GB10 memory is shared with the host,
  and the engine uses most of it.

## What `./start.sh` does

Run it on the Spark that will serve the API (rank 0). Each step is skipped when
already done, so rerunning it just restarts the engine.

1. **Image.** Pulls `ghcr.io/enntity/atlas-sparkglm:<tag>`, whose tag is the
   git tree of [`install/`](install/), so the image always matches your
   checkout. If it isn't published or `BUILD=1` is set, it builds the image
   from source instead ([`install/build.sh`](install/build.sh): Atlas at the
   commit pinned in [`install/atlas-source.json`](install/atlas-source.json),
   FlashKDA and FlashInfer's sparse-MLA prefill; 30–60 minutes cold). It then
   pulls or copies the image to the other Spark.
2. **Weights.** Downloads [`nvidia/GLM-5.3-Flash-NVFP4`](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4)
   and the [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)
   drafter at pinned revisions, and copies them to the other Spark. It uses
   `hf` when installed and a throwaway container otherwise.
3. **Overlay.** On each Spark, converts the 864 MTP matrices Atlas needs to
   NVFP4 in an overlay that links back to the original checkpoint, then
   verifies every payload (about a minute; [`install/convert.sh`](install/convert.sh)).
4. **Serve.** Starts rank 1 over ssh, then rank 0, and waits for health
   (about 2.5 minutes), then answers one smoke-test question.

Other commands: `./start.sh stop | status | logs [worker] | build | download | convert`.

## Use it

The API listens on rank 0's loopback, `http://127.0.0.1:8893/v1`, model
`glm-5.3-flash-atlas`, with no authentication. Put your own authenticated
proxy in front of it before exposing it.

```sh
curl -s http://127.0.0.1:8893/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "glm-5.3-flash-atlas", "max_tokens": 256,
  "messages": [{"role": "user", "content": "Write a haiku about two small computers."}]}'
```

Streaming, tool calls, JSON and structured output, reasoning controls
(`chat_template_kwargs.enable_thinking`, `reasoning_effort`), images and short
videos are supported.

**Profiles** (`PROFILE` in `.env`):

- `4x512k` (default): up to four requests with up to 512K context each.
- `8x128k`: up to eight requests with up to 128K each, for more concurrent
  short work.

All requests share one FP8-latent KV pool that also holds the prefix cache:
about 600K tokens at the default `GPU_MEMORY_UTILIZATION` of 0.88, and 1.39M
at 0.92 on Sparks that run nothing else. Requests that don't fit wait for
room. Prefix caching keeps the 16 most recent conversations' recurrent state
warm.

## Reproduce our numbers

The benchmark drivers are in [`bench/`](bench/). See
[docs/REPRODUCE.md](docs/REPRODUCE.md) for the exact commands and the
comparison videos.

## Other ways to run it

- **By hand:** every step of `./start.sh` as a separate command, in
  [docs/INSTALL-MANUAL.md](docs/INSTALL-MANUAL.md).
- **With [LLooM](https://github.com/Enntity/lloom):** the recipe
  `linux-nvidia-dgx-spark-2x-glm53-atlas` puts SparkGLM behind LLooM's
  authenticated gateway ([docs/LLOOM.md](docs/LLOOM.md)).

## The engine

Atlas here is [`Enntity/atlas`](https://github.com/Enntity/atlas) branch
`sparkglm/atlas-20260928`, built from three layers:

1. Atlas-Inf `main`.
2. Our GLM-5.3-Flash support and optimizations (branch `upstream/glm53-flash`),
   which we intend to propose to Atlas-Inf after review.
3. One SparkGLM-only commit adding FlashKDA and native sparse-MLA prefill
   bridges, which rely on libraries built outside the Atlas tree.

## History

SparkGLM began as a vLLM recipe building on MiaAI-Lab's two-Spark EXL3 work.
That version is preserved at the tag
[`vllm-final`](https://github.com/Enntity/sparkglm/tree/vllm-final) and the
branch `archive/vllm`, including its results and qualification records.

## License

AGPL-3.0-only ([LICENSE](LICENSE)); third-party components keep their own
licenses ([NOTICE](NOTICE), [docs/LICENSING.md](docs/LICENSING.md)). The model
weights are not distributed here. **The DFlash2 drafter is licensed CC BY-NC-ND
4.0 (non-commercial).** Check both checkpoints' terms before downloading.
