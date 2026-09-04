# SparkGLM derivation

The publication `main` contains the accepted two-GB10 engine developed from
MiaAI-Lab's recipe at `eb0469fbb2b49fd7c025f594a3339a121e58f7a9`. The original
logical commits, authors, messages, and attribution trailers are preserved as
mailbox patches under `research/current-engine-history/patches/`; the public
branch itself uses a fresh sanitized history so excluded model artifacts from
the upstream recipe are not reachable through Git history.

Before downloading weights, read `docs/LICENSING.md`. The default deliberately
matches the published four-stream video and therefore selects DFlash2 k=7.
That checkpoint is fetched separately and licensed CC BY-NC-ND 4.0, including
non-commercial/no-derivatives restrictions. Use `SPEC_METHOD=mtp` for the
checkpoint's built-in MTP path or `SPEC_METHOD=none` to disable speculation.
Model and drafter revisions are pinned independently from source and container
revisions. The exact runtime mapping and evidence boundary are in
`docs/PUBLISHED_VIDEO_CONFIGURATION.md`.

The canonical image is `sparkglm:local`, built by `./start.sh` from the root
`Dockerfile` and shipped byte-for-byte to rank 1. The root image applies the
packed-RMSNorm runtime patch itself and compiles the SM121 EXL3 extension from
the pinned ExLlamaV3 source. Its labels bind the recipe hash and SparkGLM Git
revision.

The complete vLLM source patches are also stored against Mia's exact vLLM
revision, `487ecf187d3dfe74d2cf6119a92881dba403c219`, for upstream development or
a from-source vLLM rebuild:

```bash
git clone https://github.com/vllm-project/vllm.git /path/to/vllm
git -C /path/to/vllm checkout 487ecf187d3dfe74d2cf6119a92881dba403c219
./scripts/apply-sparkglm-vllm-patches.sh /path/to/vllm
```

The native FP16 sparse-selector patch changes DeepGEMM and therefore cannot be
applied as a Python overlay. It is research-only until a source-built artifact
passes the current gates; it is not silently claimed by the canonical image.

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
