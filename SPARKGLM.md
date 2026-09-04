# SparkGLM derivation

The publication `main` contains the accepted two-GB10 engine developed from
MiaAI-Lab's recipe at `eb0469fbb2b49fd7c025f594a3339a121e58f7a9`. The original
logical commits, authors, messages, and attribution trailers are preserved as
mailbox patches under `research/current-engine-history/patches/`; the public
branch itself uses a fresh sanitized history so excluded model artifacts from
the upstream recipe are not reachable through Git history.

The EXL3 changes build through this repository's normal `Dockerfile`. The two
vLLM-core changes are stored as patches against the exact vLLM source revision
used by Mia, `487ecf187d3dfe74d2cf6119a92881dba403c219`. Apply them before building
the base `vllm-openai` image:

```bash
git clone https://github.com/vllm-project/vllm.git /path/to/vllm
git -C /path/to/vllm checkout 487ecf187d3dfe74d2cf6119a92881dba403c219
./scripts/apply-sparkglm-vllm-patches.sh /path/to/vllm
```

Build that checkout's `vllm-openai` target, then pass its image tag as
`--build-arg BASE=...` when building this recipe. This order intentionally
reuses Mia's compatibility overlay on top of the patched vLLM base.

For a fair endpoint comparison, the untouched and optimized images must use
the same model revision, DFlash revision and topology, KV/cache budget,
scheduler settings, warmup workload, prompts, arrival offsets, output length,
and alternating run order. A ready banner alone is not equivalent warm state.

For fast qualification on the existing Sparks,
`Dockerfile.sparkglm-binary` derives directly from the untouched Mia control
image and copies native artifacts from the previously qualified source build.
It then applies only the two runtime Python patches above. This avoids a second
full vLLM compile while ensuring the final filesystem starts from latest Mia,
not from the historical source-integrated image. It is a qualification path;
the patch-based source build remains the canonical reproducible path.

The reduced-precision sparse-selector implementation is present but FP16 score
output is explicit opt-in. `auto` remains FP32 because the DeepGEMM binary in
the accepted qualification image rejects FP16 logits during CUDA-graph
profiling. Do not switch the default until the exact final DeepGEMM artifact,
both TP ranks, graph capture, and the long-context semantic/throughput gates
all pass together.
