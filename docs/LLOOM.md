# Running SparkGLM under LLooM

[LLooM](https://github.com/Enntity/lloom) can run SparkGLM as one of its
managed models, behind its authenticated gateway, instead of `./start.sh`:

```sh
lloom setup --recipe linux-nvidia-dgx-spark-glm53-atlas --additive --apply --yes
lloom runtime-start glm53-flash-atlas-cluster
```

The recipe pins a SparkGLM release and uses the same image, profiles and
engine. Its current version, 7, installs SparkGLM 0.3.6 in Spark mode. Options
pick the deployment:

| Option | Values | Default |
| --- | --- | --- |
| `--sparks` | `2`: a pair (profile `4x1m`); `4`: a four-Spark chain (profile `4x1m-tp4`) | `2` |
| `--drafter` | `dflash2`: `incoai/GLM-5.3-Flash-DFlash2`; `v81` or `v80`: Enntity's drafters (private repository) | `dflash2` |
| `--memory` | `full`: 114 GB a Spark; `shared`: 108 GB, for a Spark that hosts about 10 GB of other services | `full` |
| `--streams` | `4`: four requests of up to 1M tokens; `8`: eight of up to 128K, on a pair | `4` |
| `--model` | `glm-5.3-flash`: `nvidia/GLM-5.3-Flash-FP4-Local`; `sage-s2.1-flash`: [Sage S2.1 Flash](#sage-s21-flash) | `glm-5.3-flash` |

A host keeps the options it was installed with until setup is given others.
Earlier releases install with `--recipe-version N`; LLooM's
[recipe notes](https://github.com/Enntity/lloom/blob/main/docs/recipes.md#atlas-sparkglm)
describe each version. Setup does three things on each Spark:

1. Downloads the pinned checkpoint and drafter.
2. Pulls the pinned image by digest and checks its `install/` tree label.
3. Registers the worker and leader runtimes.

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

For four nodes, pass `--sparks 4`, which uses the `4x1m-tp4` profile.
Each rank must receive the same `SPARKGLM_MODE=spark|strict` (spark when
unset). Strict defaults to serial target execution; `SPARKGLM_STRICT_MTP=1` explicitly enables the
checkpoint's native predictor while retaining the four-slot capacity and
prefix cache. The TP4 profile keeps shared-expert TP splitting off in both
modes after the Strict C4 arithmetic control. Spark remains unsplit on TP2
as well; TP2 Strict's setting is unchanged.

The [tool-call guard](TOOL-GUARD.md) is on unless a runtime sets
`SPARKGLM_TOOL_GUARD=off`. Give every rank the same `SPARKGLM_TOOL_GUARD*`
values; profiles cannot set them.

A request that turns thinking off reasons at Low effort under a 256-token
cap in Spark mode, and at Max effort with no limit, as the reference server
does, in Strict mode. `SPARKGLM_THINKING_OFF=low` or `reference` on every
rank chooses either.

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

## Sage S2.1 Flash

`--model sage-s2.1-flash` serves
[Enntity/Sage-S2.1-Flash-NVFP4](https://huggingface.co/Enntity/Sage-S2.1-Flash-NVFP4)
in place of the GLM checkpoint: a calibrated NVFP4 quantization of
`internlm/Sage-S2.1-Flash` (MIT), 179 GB, with text, image and video input.

```sh
lloom setup --recipe linux-nvidia-dgx-spark-glm53-atlas --model sage-s2.1-flash --additive --apply --yes
```

- It is served under the same gateway model, `glm-5.3-flash-atlas`, so routes
  and aliases stay as they are. To switch, suspend the model, run setup with
  the other `--model`, and resume.
- Installed this way on a pair on 2026-11-18, Sage passed 109/109 live checks
  and decoded 38.6 / 67.9 / 90.0 tok/s (prose, code, structured) against
  GLM-5.3-Flash's 37.7 / 66.1 / 88.8 on the same pair. Its prompt cache pool
  read 2.22M tokens against GLM-5.3-Flash's 2.38M in consecutive starts.
- It needs 0.3.6: on 0.3.5 Sage returned forced tool calls as text
  ([results](../results/2026-11-18-0.3.6/RESULT.md)).
- Not run: Sage on four Sparks, with `--memory shared` or with `--streams 8`.
- The values match `./start.sh --model`. The earlier `glm53-flash` and
  `sage-s21-flash` still select the same checkpoints.
