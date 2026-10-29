# SPDX-License-Identifier: AGPL-3.0-only
# Context: build.sh's pinned Enntity/atlas engine checkout and packaged install assets.
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d AS native
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    git ca-certificates build-essential python3 && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/flashinfer-ai/flashinfer.git /opt/flashinfer && \
    git -C /opt/flashinfer checkout --detach 8eccd0c1352165302840c0e19066bc42d36dbd7a
COPY engine/scripts/dev/glm_sparse_native /engine/scripts/dev/glm_sparse_native
COPY install/build-native.sh /product/install/build-native.sh
COPY install/flash_kda/ /product/install/flash_kda/
COPY install/converter/ /product/install/converter/
ENV NVCC_THREADS=2
# Candidate engine (Atlas-Inf #220 stack): it builds its GLM sparse kernels in-tree
# and loads neither the sparse bridge nor FlashKDA (it has no
# research/flash-kda-prefill), so no native library is built or shipped.
RUN mkdir -p /opt/atlas/native/notices && : > /opt/atlas/native/native-libraries.sha256
WORKDIR /product/install/converter
RUN nvcc -O3 -arch=sm_121f --fmad=false --expt-relaxed-constexpr -DTQ_PLUS_SIGNS \
    -shared -Xcompiler=-fPIC wrapper.cu -o libatlas_mtp_quantize.so && \
    PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s . -v

FROM native AS builder
RUN apt-get update -qq && apt-get install -y -qq --allow-change-held-packages \
    curl pkg-config cmake libclang-dev libibverbs-dev librdmacm-dev \
    libnccl2=2.31.2-1+cuda13.3 libnccl-dev=2.31.2-1+cuda13.3 && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --profile minimal --default-toolchain 1.93.1
ENV PATH="/root/.cargo/bin:${PATH}"
ENV RUSTUP_TOOLCHAIN=1.93.1 CUDA_HOME=/usr/local/cuda CUTLASS_HOME=/opt/cutlass
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs
RUN git clone https://github.com/NVIDIA/cutlass.git /opt/cutlass && \
    git -C /opt/cutlass checkout cf064d2e6bad2886238ac565b3b49007764f4939
WORKDIR /build
COPY engine/Cargo.toml engine/Cargo.lock engine/rust-toolchain.toml engine/README.md ./
COPY engine/crates/ crates/
COPY engine/kernels/ kernels/
COPY engine/vendor/ vendor/
# The candidate engine's GLM sparse kernels include NVIDIA's sparse-MLA prefill headers.
COPY engine/third_party/ third_party/
COPY engine/jinja-templates/ jinja-templates/
COPY engine/tests/fixtures/ tests/fixtures/
ENV ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=glm-5.3-flash ATLAS_TARGET_QUANT=nvfp4
ARG BUILD_JOBS=4
# BuildKit cache mount for the cargo target directory.
#
# Without it, iterating on engine source costs a full cold rebuild: the COPY
# above takes the whole `crates/` tree as ONE layer, so editing a single .rs
# file invalidates it, and with no cache the dependency graph plus every nvcc
# kernel is recompiled from scratch — measured at ~18 min per iteration.
# Requiring BuildKit is deliberate; the legacy builder ignores the mount, which
# would only look like the old slow behaviour rather than an error.
#
# Every cargo step below mounts the same cache (`sharing=shared`), and the
# binary is copied out of the mount because mounted paths never reach a layer.
RUN --mount=type=cache,target=/build/target,id=atlas-cargo-target,sharing=shared \
    cargo build --locked --release -j "$BUILD_JOBS" -p spark-server && \
    install -D target/release/spark /build/dist/spark
# License notices of every third-party crate compiled into the binary.
COPY install/rust-notices.py /product/install/rust-notices.py
RUN python3 /product/install/rust-notices.py /build /build/dist/notices-rust

# Release checks on the same Linux/CUDA build as the shipped binary. A stage of
# their own: the runtime image below copies only from `builder`, so BuildKit
# runs these only when build.sh asks for this target (ATLAS_IMAGE_CHECKS=1, the
# default).
FROM builder AS checks
# All native tests compile the common engine, including both model fixtures.
COPY engine/ /build/
# Check the actual compiled GLM module set and the startup vision selector.
# CUDA stubs permit these CPU-only tests without GPU build entitlements.
RUN mkdir -p /opt/cuda-stubs && \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so /opt/cuda-stubs/libcuda.so.1
RUN --mount=type=cache,target=/build/target,id=atlas-cargo-target,sharing=shared \
    LD_LIBRARY_PATH=/opt/cuda-stubs:/usr/local/cuda/lib64 \
    cargo test --locked --release -j "$BUILD_JOBS" -p atlas-kernels --lib \
    glm5_target_ships_native_vision_module -- --ignored > /tmp/atlas-kernel-check.log 2>&1 || \
    { cat /tmp/atlas-kernel-check.log; exit 1; }; \
    cat /tmp/atlas-kernel-check.log && grep -F 'test result: ok. 1 passed;' /tmp/atlas-kernel-check.log
RUN --mount=type=cache,target=/build/target,id=atlas-cargo-target,sharing=shared \
    LD_LIBRARY_PATH=/opt/cuda-stubs:/usr/local/cuda/lib64 \
    cargo test --locked --release -j "$BUILD_JOBS" -p spark-server --bin spark \
    vision_guard_ > /tmp/atlas-vision-check.log 2>&1 || \
    { cat /tmp/atlas-vision-check.log; exit 1; }; \
    cat /tmp/atlas-vision-check.log && grep -F 'test result: ok. 2 passed;' /tmp/atlas-vision-check.log

# The display carveout comes from the driver's scanout heap: the size search
# keeps the largest free range and refuses an allocator that is not the heap.
RUN --mount=type=cache,target=/build/target,id=atlas-cargo-target,sharing=shared \
    LD_LIBRARY_PATH=/opt/cuda-stubs:/usr/local/cuda/lib64 \
    cargo test --locked --release -j "$BUILD_JOBS" -p spark-server --bin spark \
    display_carveout > /tmp/atlas-carveout-check.log 2>&1 || \
    { cat /tmp/atlas-carveout-check.log; exit 1; }; \
    cat /tmp/atlas-carveout-check.log && grep -F 'test result: ok. 6 passed;' /tmp/atlas-carveout-check.log

# Exercise the serving-capacity boundary and terminal error delivery on the
# same Linux/CUDA build as the shipped binary. Require each filter to match.
# The model and runtime test fixtures (the FP4-Local configs, header extracts and CSF golden data
# the glm5 and weight-map tests embed), as SparkQwen's image copies them.
RUN --mount=type=cache,target=/build/target,id=atlas-cargo-target,sharing=shared \
    set -eu; for filter in glm_repair_policy glm_index_capacity; do \
      LD_LIBRARY_PATH=/opt/cuda-stubs:/usr/local/cuda/lib64 \
      cargo test --locked --release -j "$BUILD_JOBS" -p spark-model --lib "$filter" \
        > /tmp/atlas-model-check.log 2>&1 || { cat /tmp/atlas-model-check.log; exit 1; }; \
      cat /tmp/atlas-model-check.log; \
      grep -E 'test result: ok. [1-9][0-9]* passed;' /tmp/atlas-model-check.log; \
    done
RUN --mount=type=cache,target=/build/target,id=atlas-cargo-target,sharing=shared \
    LD_LIBRARY_PATH=/opt/cuda-stubs:/usr/local/cuda/lib64 \
    cargo test --locked --release -j "$BUILD_JOBS" -p spark-server --bin spark \
    scheduler::lifecycle_tests > /tmp/atlas-lifecycle-check.log 2>&1 || \
    { cat /tmp/atlas-lifecycle-check.log; exit 1; }; \
    cat /tmp/atlas-lifecycle-check.log && grep -E 'test result: ok. [1-9][0-9]* passed;' /tmp/atlas-lifecycle-check.log

RUN --mount=type=cache,target=/build/target,id=atlas-cargo-target,sharing=shared \
    LD_LIBRARY_PATH=/opt/cuda-stubs:/usr/local/cuda/lib64 \
    cargo test --locked --release -j "$BUILD_JOBS" -p spark-server --bin spark \
    auto_tool_eos_tests > /tmp/atlas-tool-stop-check.log 2>&1 || \
    { cat /tmp/atlas-tool-stop-check.log; exit 1; }; \
    cat /tmp/atlas-tool-stop-check.log && grep -F 'test result: ok. 7 passed;' /tmp/atlas-tool-stop-check.log

RUN --mount=type=cache,target=/build/target,id=atlas-cargo-target,sharing=shared \
    LD_LIBRARY_PATH=/opt/cuda-stubs:/usr/local/cuda/lib64 \
    cargo test --locked --release -j "$BUILD_JOBS" -p spark-server --bin spark \
    scheduler::admission::tests > /tmp/atlas-admission-check.log 2>&1 || \
    { cat /tmp/atlas-admission-check.log; exit 1; }; \
    cat /tmp/atlas-admission-check.log && grep -E 'test result: ok. [1-9][0-9]* passed;' /tmp/atlas-admission-check.log

# Buffered disconnect cancellation must retire real response receivers and
# release parked/prefilling ownership without a lifecycle reply or extra work.
RUN --mount=type=cache,target=/build/target,id=atlas-cargo-target,sharing=shared \
    set -eu; for filter in cancellation_tests disconnect_prefill_tests buffered_disconnect_; do \
      LD_LIBRARY_PATH=/opt/cuda-stubs:/usr/local/cuda/lib64 \
      timeout --signal=TERM --kill-after=10s 180 \
      cargo test --locked --release -j "$BUILD_JOBS" -p spark-server --bin spark "$filter" \
        > /tmp/atlas-cancel-check.log 2>&1 || { cat /tmp/atlas-cancel-check.log; exit 1; }; \
      cat /tmp/atlas-cancel-check.log; \
      grep -E 'test result: ok. [1-9][0-9]* passed;' /tmp/atlas-cancel-check.log; \
    done

# Native Poolside streaming headers and arguments must survive nested JSON
# fields and arbitrary chunks. Exercise all tool-parser formats in the binary.
RUN --mount=type=cache,target=/build/target,id=atlas-cargo-target,sharing=shared \
    LD_LIBRARY_PATH=/opt/cuda-stubs:/usr/local/cuda/lib64 \
    cargo test --locked --release -j "$BUILD_JOBS" -p spark-server --bin spark \
    tool_parser:: > /tmp/atlas-parser-check.log 2>&1 || \
    { cat /tmp/atlas-parser-check.log; exit 1; }; \
    cat /tmp/atlas-parser-check.log && grep -E 'test result: ok. [1-9][0-9]* passed;' /tmp/atlas-parser-check.log

# DFlash split-head verification must preserve the requested distribution,
# including whole-row fallback and seeded serial/speculative equivalence.
# Explicit thinking caps still close reasoning; a zero server ceiling stays
# uncapped on spontaneous entry and re-entry. Completions expose original
# generated token IDs only when requested. Request-level disable_mtp must
# reach both handlers and override speculative dispatch. Each filter executes tests.
RUN --mount=type=cache,target=/build/target,id=atlas-cargo-target,sharing=shared \
    set -eu; for filter in dflash_seq_policy verify_dflash_step::think_end \
                          verify_dflash_step::split_head verify_dflash_step::masked_path_tests \
                          verify_pipeline_helper::split_rows verify_pipeline_helper::scored::tests prediction_usage unscored_logprob_admission batched_logprob_read_failure_aborts_every_advanced_member spontaneous_thinking_ \
                          glm_prompt_parity_tests completion_token_ids_ \
                          completion_disable_mtp_reaches_scheduler \
                          request_disable_mtp_overrides_all_spec_dispatch_modes; do \
      LD_LIBRARY_PATH=/opt/cuda-stubs:/usr/local/cuda/lib64 \
      cargo test --locked --release -j "$BUILD_JOBS" -p spark-server --bin spark "$filter" \
        > /tmp/atlas-verify-check.log 2>&1 || { cat /tmp/atlas-verify-check.log; exit 1; }; \
      cat /tmp/atlas-verify-check.log; \
      grep -E 'test result: ok. [1-9][0-9]* passed;' /tmp/atlas-verify-check.log; \
    done
RUN --mount=type=cache,target=/build/target,id=atlas-cargo-target,sharing=shared \
    set -eu; for check in spark-runtime:sampler::tests spark-model:glm_verify_sample; do \
      LD_LIBRARY_PATH=/opt/cuda-stubs:/usr/local/cuda/lib64 \
      cargo test --locked --release -j "$BUILD_JOBS" -p "${check%%:*}" --lib "${check#*:}" \
        > /tmp/atlas-sample-check.log 2>&1 || { cat /tmp/atlas-sample-check.log; exit 1; }; \
      cat /tmp/atlas-sample-check.log; \
      grep -E 'test result: ok. [1-9][0-9]* passed;' /tmp/atlas-sample-check.log; \
    done

# Refusal content: keep the full blocking response body alongside refusal
# metadata, and cover the existing streaming content contract.
RUN --mount=type=cache,target=/build/target,id=atlas-cargo-target,sharing=shared \
    set -eu; for filter in refusal api::chat_stream; do \
      LD_LIBRARY_PATH=/opt/cuda-stubs:/usr/local/cuda/lib64 \
      cargo test --locked --release -j "$BUILD_JOBS" -p spark-server --bin spark "$filter" \
        > /tmp/atlas-refusal-check.log 2>&1 || { cat /tmp/atlas-refusal-check.log; exit 1; }; \
      cat /tmp/atlas-refusal-check.log; \
      grep -E 'test result: ok. [1-9][0-9]* passed;' /tmp/atlas-refusal-check.log; \
    done

# Vision prefix cache: content keys (different images miss, a re-sent image
# hits, ranks agree), the rank-1 receive-only tower, the per-chunk encode and
# its stream ordering, and the hash/deferred fields on the rank wire.
RUN --mount=type=cache,target=/build/target,id=atlas-cargo-target,sharing=shared \
    set -eu; for check in spark-runtime:vision spark-model:vision_cache spark-model:vision_transport \
                          spark-model:vision_encoder spark-model:embed_chunk; do \
      LD_LIBRARY_PATH=/opt/cuda-stubs:/usr/local/cuda/lib64 \
      cargo test --locked --release -j "$BUILD_JOBS" -p "${check%%:*}" --lib "${check#*:}" \
        > /tmp/atlas-vision-cache-check.log 2>&1 || { cat /tmp/atlas-vision-cache-check.log; exit 1; }; \
      cat /tmp/atlas-vision-cache-check.log; \
      grep -E 'test result: ok. [1-9][0-9]* passed;' /tmp/atlas-vision-cache-check.log; \
    done

# The common engine's serving and media contracts must run for the GLM target
# too. Keep the complete pinned checkout in this build-only stage because
# workspace tests inspect fixtures, source and committed benchmark definitions.
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/build/target,id=atlas-cargo-target,sharing=shared \
    LD_LIBRARY_PATH=/opt/cuda-stubs:/usr/local/cuda/lib64 \
    cargo test --locked --workspace --release --no-fail-fast -j "$BUILD_JOBS"

# Run the workspace lint gate with the same pinned Rust compiler as the image.
RUN --mount=type=cache,target=/build/target,id=atlas-cargo-target,sharing=shared \
    rustup component add clippy --toolchain 1.93.1 && \
    ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 \
    cargo clippy --locked --workspace --tests -j "$BUILD_JOBS"

FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04@sha256:95318efecfd68ab3d109da5277863257b06137c84f34a87f38de970d5cd035d3 AS runtime
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends --allow-change-held-packages \
    python3 ca-certificates ffmpeg libnccl2=2.31.2-1+cuda13.3 libibverbs1 librdmacm1 \
    ibverbs-providers libnl-3-200 libnl-route-3-200 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/dist/spark /usr/local/bin/spark
COPY --from=builder /build/jinja-templates/ /jinja-templates/
COPY --from=native /opt/atlas/native/native-libraries.sha256 /opt/atlas/native/
COPY --from=native /opt/atlas/native/notices/ /opt/atlas/notices/
COPY --from=builder /opt/cutlass/LICENSE.txt /opt/atlas/notices/CUTLASS-LICENSE.txt
COPY --from=builder /build/dist/notices-rust/ /opt/atlas/notices/rust/
COPY --from=native /product/install/converter/ /opt/atlas/converter/
COPY install/serve.py install/release.json /opt/atlas/
COPY install/profiles/ /opt/atlas/profiles/
COPY install/atlas-source.json /opt/atlas/source-manifest.json
COPY engine/LICENSE /LICENSE
ENV LD_LIBRARY_PATH=/opt/atlas/native:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64
ENV RUST_LOG=info
EXPOSE 8893
# The image's identity is the git tree of install/ (its tag); label it last.
ARG INSTALL_TREE
ARG PRODUCT_VERSION
RUN python3 -c 'import json,sys; assert json.load(open("/opt/atlas/release.json"))["version"] == sys.argv[1]' "$PRODUCT_VERSION"
RUN test -n "$INSTALL_TREE"
LABEL org.opencontainers.image.version="$PRODUCT_VERSION" \
      org.opencontainers.image.licenses="AGPL-3.0-only" \
      org.opencontainers.image.source="https://github.com/Enntity/sparkglm" \
      io.enntity.sparkglm.install-tree="$INSTALL_TREE"
ENTRYPOINT ["python3", "/opt/atlas/serve.py"]
