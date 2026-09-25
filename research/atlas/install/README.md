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

The candidate launch profile requests a 262,144-token total sequence budget
from a shared 270,336-token BF16 KV pool. Four ordinary text requests can use
the speculative decoder within its 32K domain. Longer requests, tools and
multimodal work use the serial native path. Admission reserves each request's
prompt, output budget and speculative spill; work that does not currently fit
queues. This does not promise four simultaneous full context windows.

Memory utilization is 0.95, with KV overcommit disabled and the 4096 MiB
free-memory guard active. The pool is capped only after validating that its
physical allocation fits the measured budget. The image includes vision weights
and preallocated scratch. The 262K profile is pending live qualification; 512K
is not advertised as supported by this Atlas build.

A stock LLooM installation can have a stricter memory reserve than this recipe
needs. Preview and explicitly configure numeric policy on each participating
Spark before starting the model:

```sh
lloom runtime-policy --max-memory-utilization 0.97 --reserve-memory-gb 4
lloom runtime-policy --max-memory-utilization 0.97 --reserve-memory-gb 4 --apply --yes
```

These candidate values require live memory qualification. This command preserves
the existing enforcement mode and per-node overrides; inspect its report and
keep normal memory enforcement enabled. It does not enable YOLO mode.

Prompts plus requested output must fit the total context. The gateway allows
output budgets up to 131,072 tokens with a four-hour request limit; this is not
a claim of full-length generation endurance. Clients can choose shorter
per-request deadlines. Large images are resized to the native encoder capacity,
and video decoding is capped at 32 frames. Aggregate visual output capacity
is separate from the language model's context window.

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
