# Install Atlas SparkGLM

This is the source-built Atlas candidate for GLM-5.3-Flash on two directly
connected NVIDIA DGX Sparks. It is an opt-in LLooM recipe. The vLLM recipe
remains the recommended default while Atlas completes appliance qualification.

With a LLooM version containing the Atlas recipe, run on the leader:

```sh
lloom setup --recipe linux-nvidia-dgx-spark-2x-glm53-atlas --additive
lloom setup --recipe linux-nvidia-dgx-spark-2x-glm53-atlas --additive --apply --yes
lloom runtime-start glm53-flash-atlas-cluster
```

Review the first command's plan, including both nodes and their fabric
addresses. Setup downloads the pinned NVIDIA checkpoint, builds the native
engine image from pinned sources, converts the MTP overlay once on each node,
and verifies the conversion. The runtime-start command then starts the worker
and leader through the owner gateway and waits for readiness. Docker,
the NVIDIA container runtime and working RoCE connectivity are prerequisites.
Allow time and disk space for a first source build and model download.
Existing model files can be reused through LLooM's managed model directory.

Use the authenticated LLooM OpenAI-compatible API with model
`glm-5.3-flash-atlas`. Backend ports stay on loopback. Model startup does not
compile code or convert weights; those operations belong to setup.

The candidate includes JSON schema/JSON output, structured function calls,
streaming, reasoning controls, images and video. Constrained generation and
multimodal requests use native decoding; ordinary text retains speculative
MTP2 decoding. FFmpeg is included and enabled for MP4/WebM clips; animated
GIF also works. Public HTTP(S) media URLs and base64 data URIs are supported.
URL fetching retains private-address, redirect, byte-size and time limits.
The scheduler separates these paths to preserve distributed
ordering. Supported behavior and hardware results must be read with the
associated qualification record; a successful source build alone is not a
capability or performance result.

The frozen launch profile uses a 36,864-token total sequence budget, four
text sequences, BF16 KV and FP32 recurrent state. The memory utilization is
0.921 to include the enabled vision tower and its preallocated workspace;
KV overcommit stays disabled and the 4096 MiB free-memory guard remains active. Prompts plus requested output
must fit that sequence budget. Vision resolution is bounded by the native
encoder allocation; large images are resized before encoding. These limits
are independent of the vLLM recipe's context and vision capacity.

For a source-only build from a clean SparkGLM checkout on a Spark:

```sh
bash research/atlas/install/build.sh /absolute/path/to/atlas-install
```

This builds an image and writes a receipt without starting a model. The
receipt binds the host's image ID to the SparkGLM commit and reconstructed
engine manifest. LLooM checks it again during setup. Runtime assets and engine
source under this directory remain AGPL-3.0-only; third-party dependency
notices are included in the image.

The serving profile explicitly selects `poolside_v1`, which matches the checkpoint’s `<tool_call>name<arg_key>…</arg_key><arg_value>…</arg_value></tool_call>` syntax. This works without a checkpoint-local `MODEL.toml`.
