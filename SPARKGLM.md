# Run SparkGLM on two DGX Sparks

This is the **current serving recipe**, not the archived Atlas experiment.
Run the commands below on the **head Spark** unless stated otherwise.

The default is the posted-video profile: GLM-5.3-Flash EXL3, TP2, mixed
scheduling, grouped prefill, cooperative decode, and DFlash2 k=7. This is a
research preview, not a G5-qualified production release.

## 1. Check the requirements

- Two DGX Spark GB10 machines, connected over CX7/RoCE, with their GPUs and
  unified memory available. Stop other full-model servers through their own
  management system first.
- Docker with NVIDIA GPU support, usable without sudo, on both machines.
- Key-based SSH from head to worker; Git, Python 3, curl, rsync, and the
  Hugging Face CLI (`hf`) on the head.
- At least 180 GiB free per node for model downloads alone; leave additional
  room for the roughly 21 GB runtime image, native build intermediates, and
  caches. The first installation builds native code locally. It is not a
  quick prebuilt-image install.
- **Check [licenses](docs/LICENSING.md) before downloading.** The separately
  fetched default DFlash2 checkpoint is CC BY-NC-ND 4.0
  (non-commercial/no-derivatives). Set `SPEC_METHOD=mtp` or `none` in
  `.env` before the first launch if those terms do not fit. Those profiles
  do not inherit the video's performance. The target/quant have separate terms.
- The target EXL3/TR3 quant is **ShapleyMCG source-available**, with attribution
  requirements for published results and named-party/channel exclusions.
  Disabling DFlash2 does not remove those terms. Read the
  [quant notice and citation](docs/QUANT_ATTRIBUTION.md) before benchmarking.
- Read [known limitations](docs/KNOWN_LIMITATIONS.md), particularly the
  sparse-MLA candidate-set approximation. We do not claim exact inference.

There is currently **no published, qualified SparkGLM prebuilt image**.
Do not substitute Mia's image and assume it contains SparkGLM's changes.
[Image-publication status](docs/IMAGE_RELEASE.md) describes that separate gate.

## 2. Clone and configure

```bash
git clone https://github.com/Enntity/sparkglm.git
cd sparkglm
cp .env.example .env
```

If `hf` is missing, install it in a local environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install huggingface_hub
hf --help
```

Edit `.env`; the example is a template, not universal network configuration:

| Setting | What to supply |
| --- | --- |
| `HEAD_IP`, `WORKER_IP` | Reachable addresses assigned to your two Sparks |
| `WORKER_USER` | Worker account, if different from the head account |
| `HEAD_CX7_IF`, `WORKER_CX7_IF` | Each node's actual CX7 network interface |
| `HEAD_CX7_IB`, `WORKER_CX7_IB` | Each node's actual RoCE HCA |
| `HEAD_GID`, `WORKER_GID` | Each HCA's populated RoCEv2 GID index, if different from `NCCL_IB_GID_INDEX` |

On **each node**, inspect interfaces with:

```bash
ip -br a
ls /sys/class/infiniband
docker info
nvidia-smi
```

For the chosen HCA, inspect its `ports/1/gids/` files under
`/sys/class/infiniband/`. Choose the entry corresponding to that node's
fabric IP; an all-zero entry is not usable. The launcher's preflight validates
the selections and prints the GID tables on failure. Verify key-based SSH
to the worker before launch. Do not copy another kit's interface names blindly.

Keep `IMAGE=sparkglm:local` and the performance settings unchanged for the
reference profile. Keep `.env` private. Existing `.env` files are preserved
across upgrades, so compare them with `.env.example` rather than overwriting
them.

### Coming from Mia or an older SparkGLM checkout?

**Both recipes currently use `glm53-exl3-head` and `glm53-exl3-worker`.**
Our `stop`/`restart` targets those names; changing only the API port does not
make the deployments independent.

1. Save the old recipe revision, private configuration, and image identity.
2. Stop the old deployment using its original launcher or fleet manager.
3. Create a fresh SparkGLM `.env`. Transfer network/account settings and,
   when useful, the existing model-cache location—not the old image or
   performance defaults.
4. Launch SparkGLM using this guide. The pinned matching weights can be reused;
   do not delete the cache.
5. To roll back, stop SparkGLM, then start the preserved old recipe/configuration.

Do not run both full models simultaneously on the pair. For already cached
`brandonmusic` weights, see `MODEL_CACHE_NAME` in `.env.example` to avoid
a duplicate download.

## 3. Build and launch

```bash
./start.sh
```

A fresh checkout builds the root Dockerfile automatically. `BUILD=1 ./start.sh`
forces a rebuild. The launcher checks both hosts, builds the image, sends the
same image to rank 1, downloads and syncs missing pinned weights, then starts
TP2. Do not use `SKIP_BUILD=1` to bypass an unbuilt or mismatched reference.

Stop resident models before compilation: the default build guard requires
at least 32 GiB available unified RAM. First build, download, weight loading,
graph capture, and initial JIT can all take time. Subsequent starts reuse
matching images, weights, and persistent JIT caches.

Progress and troubleshooting:

| Stage | Where to look / what to expect |
| --- | --- |
| Native compilation | `tail -f logs/build-sm121.log`; no model is serving yet |
| Download/sync | Launcher output; `./download.sh` optionally stages weights on the head only |
| Weight load/graph capture | `./start.sh logs` and `./start.sh logs worker` |
| HTTP healthy, still warming | Short shape warmup, then mandatory four-stream 16K capacity check |
| Fully ready | Wait for the launcher's final `is UP` banner, not just `/health` |

The long capacity check must pass for the default four-stream profile.
Failure is not proof of a bad model: inspect worker logs, available memory,
cache capacity, and networking. Do not skip it merely to claim a healthy
four-stream appliance. A failed warmup may leave containers running; inspect
them with `./start.sh status`, then use `./start.sh stop` if abandoning the
attempt.

## 4. Get a streamed answer

The default API is head-local at `http://127.0.0.1:8888/v1`, with model ID
`GLM-5.3-Flash-EXL3`. On the head:

```bash
curl --fail --no-buffer http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "GLM-5.3-Flash-EXL3",
    "messages": [{"role": "user", "content": "Explain a hash table in three sentences."}],
    "stream": true,
    "max_tokens": 256,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

You should see SSE `data:` events, followed by `[DONE]`. These are raw
streaming API events, not a chat UI. This is a functional smoke test, not a
throughput benchmark.

If you set `VLLM_API_KEY`, add an `Authorization: Bearer` header using that
key. A custom `PORT` changes the example URL.

To use the endpoint from a laptop, prefer an SSH tunnel to the head while
keeping the API on loopback. Alternatively, set `API_HOST=0.0.0.0` **and**
configure `VLLM_API_KEY`, then restart. Use that wildcard rather than binding
only a LAN address: the current launcher's readiness probes use loopback.
Protect the
host-networked TP/RoCE fabric and diagnostic endpoints from untrusted clients;
API authentication is not a firewall.

## Everyday commands

```bash
./start.sh status
./start.sh logs
./start.sh logs worker
./start.sh stop
./start.sh restart
```

Restart after configuration changes. Native compilation is needed only when
the recipe/image changes, not on every start. Avoid skip flags until you
understand which checks or transfers they bypass.

## Where to go next

- [Contribute or run tinyGLM](CONTRIBUTING.md).
- [Exact video profile and source identity](docs/PUBLISHED_VIDEO_CONFIGURATION.md).
- [Evidence and limitations](results/CURRENT.md).
- [Credits](docs/ATTRIBUTION.md) and [source provenance](docs/PROVENANCE.md).
- [Historical Mia guide](docs/upstream/MIA_RECIPE_README.md), retained for
  upstream credit and reference, not a second SparkGLM installation procedure.
