# Running SparkGLM under LLooM

[LLooM](https://github.com/Enntity/lloom) can run SparkGLM as one of its
managed models, behind its authenticated gateway, instead of `./start.sh`:

```sh
lloom setup --recipe linux-nvidia-dgx-spark-2x-glm53-atlas --additive --apply --yes
lloom runtime-start glm53-flash-atlas-cluster
```

The recipe pins a SparkGLM commit and uses the same image, profiles and
engine. Setup does four things on each Spark:

1. Downloads the pinned checkpoints.
2. Gets the image for the pinned `install/` tree, pulling it or building it.
3. Converts and verifies the MTP overlay, for recipes that pin the NVFP4
   checkpoint (FP4-Local, this repository's default since 2026-10-17, needs none).
4. Registers the worker and leader runtimes.

Starting launches rank 1, then rank 0, and routes the gateway model
`glm-5.3-flash-atlas` to the leader.

LLooM doesn't set host options, so cap the GPU clock on both Sparks first:
`sudo bash gpu-clock-cap.sh` from a SparkGLM checkout on each (see the
README's requirements).

The recipe is additive: it doesn't change your default model or aliases.
Nothing is built or converted while a runtime starts.

## Unified TP2/TP4 release

The unified release uses one SparkGLM image and the same precision policies at
both sizes. See [current validation and image identity](../results/2026-10-21-unified-rc/RESULT.md).
An existing LLooM recipe may pin an earlier SparkGLM commit; check its resolved image and
`install/` tree before assuming these options are present.

For four nodes, the recipe is
`linux-nvidia-dgx-spark-4x-glm53-atlas-tp4`, using the `4x1m-tp4` profile.
Each rank must receive the same `SPARKGLM_MODE=strict|spark`. Strict defaults
to serial target execution; `SPARKGLM_STRICT_MTP=1` explicitly enables the
checkpoint's native predictor while retaining the four-slot capacity and
prefix cache. The TP4 profile keeps shared-expert TP splitting off in both
modes after the Strict C4 arithmetic control. Spark remains unsplit on TP2
as well; TP2 Strict's setting is unchanged.

The [tool-call guard](TOOL-GUARD.md) is on unless a runtime sets
`SPARKGLM_TOOL_GUARD=off`. Give every rank the same `SPARKGLM_TOOL_GUARD*`
values; profiles cannot set them.

Four-node topology defaults to `SPARKGLM_TOPOLOGY=chain`. To use a ring,
configure `SPARKGLM_TOPOLOGY=ring` on every runtime and supply each node's
`FABRIC_LEFT_HCA` and `FABRIC_RIGHT_HCA` lists, including the closing 3–0
edge. Both rails of a dual-rail ring must continue across that edge.
`FABRIC_INTERFACE` selects the socket network. These settings must describe
working host networking; selecting ring does not configure host routes.
See the [per-rank map and launcher contract](INSTALL-MANUAL.md#four-spark-chain-or-ring).

Cache namespaces separate TP2 from TP4, chain from ring, and Strict from
Spark. The engine now supports TP4 sharded KV spill/restore through the NVMe
prefix tier; the per-node launcher uses the same `--prefix-cache-dir` option.
Strict smoke passed on chain and ring; separate checks passed near-1M
shared-prefix C4 capacity and two disk eviction/restore cycles. All release
profiles enable `ATLAS_GLM_PC_WRITE_FLOOR=1` to protect matched prefix blocks
during another request's recomputation. Use a single image identity across
all ranks and check the resolved mode/topology in startup receipts.
