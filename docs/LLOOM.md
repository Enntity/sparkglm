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
3. Converts and verifies the overlay.
4. Registers the worker and leader runtimes.

Starting launches rank 1, then rank 0, and routes the gateway model
`glm-5.3-flash-atlas` to the leader.

The recipe is additive: it doesn't change your default model or aliases.
Nothing is built or converted while a runtime starts.
