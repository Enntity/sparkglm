# Optional adaptive verification and SSD profile

This publishes the source and settings of the privately measured NVIDIA NVFP4 TP2 appliance. It is an opt-in AGPL derivative, separate from the default Apache serving path. Publishing these bytes does not qualify a newly built image or promote a new default.

The target is NVIDIA GLM-5.3-Flash NVFP4 revision `423acf37583782c51c142d145aef733d72943d93`, with MXFP8 DFlash2 revision `610aa967a92bfeb97e3d848dcb8693553e8b6a55`. The profile uses TP2, draft K5 with adaptive verification2/4/5, compact draft pages256, eight sequence slots,262144 context,11GiB KV per rank,2048-token chunks, and unthrottled mixed prefill. The optional entrypoint retains the measured24 capture sizes, producing query lengths3/5/6. Eight slots do not promise eight full262K contexts.

## Build and configure

Use an already qualified NVIDIA NVFP4/MXFP8 serving base with the compact-KV patch. This is a derivative of that exact composition, not a claim that any generic vLLM image works. `source-contract.json` pins both pre-adaptive source files and the exact measured output. The installer verifies both inputs, stages and compiles both patched outputs, then installs them. An incompatible or partly patched base is rejected without touching either installed file. Run this during image construction only.

From the repository root, on each node:

```sh
docker build -f research/adaptive-verification/Dockerfile \
  --build-arg BASE="$QUALIFIED_BASE_IMAGE" -t sparkglm-adaptive:local .
docker build -f Dockerfile.nvme-prefix-cache \
  --build-arg BASE=sparkglm-adaptive:local -t sparkglm-adaptive-ssd:local .
docker image inspect --format '{{.Id}}' sparkglm-adaptive-ssd:local
```

Apply adaptive **before** the SSD layer: the SSD scheduler patch expects the measured adaptive source. Keep each resulting immutable image ID. These local builds are not GPU-tested by this publication. The build carries the complete adaptive, vLLM and launcher source/notices; source distribution obligations apply to the derivative.

Render the LLooM recipe locally with the two image IDs:

```sh
python3 research/adaptive-verification/render_profile.py \
  --head-image "$HEAD_IMAGE_ID" --worker-image "$WORKER_IMAGE_ID" \
  --output /tmp/sparkglm-adaptive-ssd.json
```

Import the rendered file using LLooM's normal recipe workflow. The source template intentionally contains `RENDER_REQUIRED` and must not be installed directly. It neither selects a default alias nor starts a service. The recipe invokes the image's `/opt/sparkglm/adaptive/entrypoint.sh`, avoiding a stale host-mounted entrypoint. That entrypoint contains the current SSD connector arguments and all ordinary SparkGLM runtime hooks.

The `sparkglm-private-nvme` Docker volume is separate on each node. Ensure Docker's volume storage is on the intended SSD, or replace that mount with a private local SSD bind mount on each node before installation. Both ranks receive the same namespace, derived from image IDs, model pins and complete recipe settings. Regenerate it when any model, serving or KV identity changes. Never copy a namespace from an unrelated build. SSD cap and free-space reserve are each64GiB per rank; I/O uses two threads. Cache files are prompt-derived private data and must never enter source or benchmark bundles. See [SSD behavior and failure recovery](../../docs/nvme-prefix-cache.md).

Adaptive can be disabled without reloading by the upstream override file `/root/.cache/vllm/glm53_adaptive_k.json`: write `{"mode":"off"}` in the head's persistent vLLM cache volume. EMA mode must be enabled at boot to create its graphs; `{"mode":"ema"}` restores it. The patch checks mtime every50 scheduling steps. An idle server may not observe the change immediately. Stale override files can supersede the environment; inspect/remove them before measuring a new profile.

## Evidence and limits

[Adaptive comparison](../../results/candidates/adaptive-verification-20260920/RESULT.md) retains the favorable C2 median and the C4/C8 regressions. The tiny-model run had speculation disabled and was inconclusive for adaptive qualification. Bounded full-model functionality passed; G5/endurance did not run. [Coding comparison](../../results/candidates/coding-agent-repeat3-20260921/RESULT.md) favors stock Mia on this small task, with quality omissions retained on both sides. Neither result proves a universal winner.

Run `python3 research/adaptive-verification/check_local.py` and `python3 tests/test_adaptive_package.py` for local-only source/policy/packaging checks. `scripts/check.sh all` includes both. They do not exercise GPUs or certify numerical equivalence.

## Source and licenses

`vendor/mia/patch_adaptive_k.py` is verbatim Mia source at `775a58b704924b13cf5c38de97559b3630f67bbf`; its full AGPL license and MIT notice are retained beside it. Vanilla fixtures originate at vLLM `487ecf187d3dfe74d2cf6119a92881dba403c219` under Apache-2.0. Installed baseline fixtures include the existing MIT recipe scheduler adaptation; patched receipts also include the AGPL adaptive changes. The entrypoint retains its MIT/Apache notices. The hash-bound installer, renderer and packaging are original AGPL integration work. See [the provenance ledger](../../provenance/upstreams.json) and [model/quant attribution](../../docs/QUANT_ATTRIBUTION.md). No weights or compiled images are distributed here.
