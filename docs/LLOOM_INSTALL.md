# Optional LLooM integration for SparkGLM

This opt-in integration requires an existing LLooM installation. SparkGLM itself is independent; see [standalone installation](../SPARKGLM.md).

The selected profile is the maintainer-selected **NVFP4 research preview**:
512K context (524,288 tokens), 9 GiB KV per rank, native FlashInfer CUTLASS,
MXFP8 DFlash2 TP2 with seven draft tokens, 2K prefill chunks, four active
sequences, and mixed scheduling. This is the configuration recorded in the
September 8 mixed video. `--skip` reproduces the other video setting.

This is a source install. No prebuilt image or complete G5 endurance/general
quality certification is claimed. See [results](../results/CURRENT.md),
[limitations](KNOWN_LIMITATIONS.md), and [model licenses](LICENSING.md).

## Prepare the cluster

Use two Linux ARM64 DGX Sparks with working NVIDIA Container Toolkit, Docker,
Python 3 with venv, Git, rsync, and passwordless SSH from leader to worker.
Use the same user/home and LLooM install path on both nodes. Configure LLooM's
[direct two-node cluster](https://github.com/Enntity/lloom/blob/main/docs/clusters.md)
and verify its fabric addresses, interface, and RoCE mapping before installing.
This launcher uses that existing cluster configuration; it does not invent
network settings or change application aliases.

Install LLooM from source on **both** nodes at the revision in
[`profiles/build.json`](../profiles/build.json), using its documented `npm ci`
and `npm link` steps. For example, on each node:

```bash
git clone https://github.com/Enntity/lloom.git
cd lloom
git checkout 2cc2f0df9ddcb1bb7fe60f7bd6934d2a9de1e4f2
npm ci
npm link
```

Configure and run its gateway service following the LLooM instructions. The launcher checks the SparkGLM entrypoint hash on both
nodes. LLooM owns admission, worker-first startup, readiness, routing and stop.

Allow at least 32 GiB **MemAvailable** on the leader for native compilation:
stop resident full models through LLooM first. Reserve ample disk for roughly
200 GB of target weights, the draft, Docker source/build layers, and the
second local weight copy on the worker; 190 GB is not enough for a fresh build.
Weights remain separately downloaded, pinned publisher artifacts.

## Install and start

Run on the Spark leader:

```bash
git clone https://github.com/Enntity/sparkglm.git
cd sparkglm
./start.sh --lloom plan
./start.sh --lloom --worker USER@WORKER
```

Replace `USER@WORKER` with the SSH destination. `--model-root /path/to/models`
selects the same absolute directory on both nodes; the default is
`~/.lloom/models`. `--lloom-root /path/to/lloom` overrides executable-based
installation discovery.

The installer builds the pinned source layers in order, copies the immutable
image to the worker, downloads pinned models through LLooM, copies those local
weights to the worker, installs an additive managed recipe, and starts it.
Existing models/default aliases remain registered. Reinstalling stops only
the selected SparkGLM runtime before replacing its configuration; clients using it will be interrupted.
A successful install ends with LLooM runtime status. Call the configured LLooM
gateway using model **`sparkglm-nvfp4`** and your gateway authentication.

For an image you already built and qualified, avoid rebuilding with:

```bash
./start.sh --lloom --worker USER@WORKER --image sha256:YOUR_FULL_IMAGE_ID
```

Both rank identities are checked. For separately built and qualified rank images, add `--worker-image sha256:WORKER_IMAGE_ID`. Use the `check` command with these image arguments to validate the installed adapter, image identities and LLooM setup plan without downloading weights or changing runtime configuration. A different image is your own experiment,
not automatically the recorded source. To reproduce skip scheduling, add
`--skip` to the install command. It retains our existing 3584-token remaining
prefill bypass and zero max-wait setting. Changing scheduling requires reinstall;
`start` simply starts the already installed profile.

```bash
./start.sh --lloom status
./start.sh --lloom stop
./start.sh --lloom start
```

The build pins retain the measured EXL3 foundation and intermediate adapter
layers, then add the measured NVFP4/MXFP8 support. E3 remains inactive for
NVFP4. Public source snapshots preserve original tree bytes; the
[revision map](../provenance/2026-09-08-publication-map.json) connects them to
historical measurement SHAs. Rebuilding is not a claim of bit-identical images.

## EXL3 and research

The latest EXL3 work is preserved on the
[`exl3` branch](https://github.com/Enntity/sparkglm/tree/exl3), including the
corrected concurrent E3 policy, 32-row threshold, and 1M profile.
Use `./start.sh --lloom --profile exl3 --worker USER@WORKER` from `main` to build and install that latest EXL3 profile through LLooM. It has its own runtime identity; stop the active full model first. The old standalone launcher is retained
as `start-exl3.sh`; it reproduces the older video foundation, **not** the latest
E3 profile. Its historical instructions are in
[EXL3_QUICKSTART.md](EXL3_QUICKSTART.md).

For experiments, use [METHODOLOGY.md](METHODOLOGY.md): model-free operator
checks, tinyGLM integration, matched full-model workloads, and semantic checks.
Do not equate synthetic fixture success with model quality or capacity.
