# SPDX-License-Identifier: AGPL-3.0-only
# Context: build.sh's verified engine reconstruction and packaged install assets.
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d AS native
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    git ca-certificates build-essential python3 && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/flashinfer-ai/flashinfer.git /opt/flashinfer && \
    git -C /opt/flashinfer checkout --detach 8eccd0c1352165302840c0e19066bc42d36dbd7a
COPY engine/scripts/dev/glm_sparse_native /engine/scripts/dev/glm_sparse_native
COPY engine/research/flash-kda-prefill /engine/research/flash-kda-prefill
COPY product/research/atlas/install/build-native.sh /product/research/atlas/install/build-native.sh
COPY product/research/atlas/flash_kda/ /product/research/atlas/flash_kda/
COPY product/research/atlas/nvidia-mtp-converter/ /product/research/atlas/nvidia-mtp-converter/
ENV NVCC_THREADS=2
RUN bash /product/research/atlas/install/build-native.sh /engine /opt/flashinfer /product /opt/atlas/native
WORKDIR /product/research/atlas/nvidia-mtp-converter
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
COPY engine/jinja-templates/ jinja-templates/
COPY engine/tests/fixtures/ tests/fixtures/
ENV ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=glm-5.3-flash-nvfp4 ATLAS_TARGET_QUANT=nvfp4
ARG BUILD_JOBS=4
RUN cargo build --locked --release -j "$BUILD_JOBS" -p spark-server
# Check the actual compiled GLM module set and the startup vision selector.
# CUDA stubs permit these CPU-only tests without GPU build entitlements.
RUN mkdir -p /opt/cuda-stubs && \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so /opt/cuda-stubs/libcuda.so.1
RUN LD_LIBRARY_PATH=/opt/cuda-stubs:/usr/local/cuda/lib64 \
    cargo test --locked --release -j "$BUILD_JOBS" -p atlas-kernels --lib \
    glm5_target_ships_native_vision_module -- --ignored > /tmp/atlas-kernel-check.log 2>&1 || \
    { cat /tmp/atlas-kernel-check.log; exit 1; }; \
    cat /tmp/atlas-kernel-check.log && grep -F 'test result: ok. 1 passed;' /tmp/atlas-kernel-check.log
RUN LD_LIBRARY_PATH=/opt/cuda-stubs:/usr/local/cuda/lib64 \
    cargo test --locked --release -j "$BUILD_JOBS" -p spark-server --bin spark \
    vision_guard_ > /tmp/atlas-vision-check.log 2>&1 || \
    { cat /tmp/atlas-vision-check.log; exit 1; }; \
    cat /tmp/atlas-vision-check.log && grep -F 'test result: ok. 2 passed;' /tmp/atlas-vision-check.log

# Exercise the serving-capacity boundary and terminal error delivery on the
# same Linux/CUDA build as the shipped binary. Require each filter to match.
COPY engine/test_data/lightning_dspark_config.json test_data/lightning_dspark_config.json
RUN set -eu; for filter in glm_repair_policy glm_pair_plan glm_mtp_capacity; do \
      LD_LIBRARY_PATH=/opt/cuda-stubs:/usr/local/cuda/lib64 \
      cargo test --locked --release -j "$BUILD_JOBS" -p spark-model --lib "$filter" \
        > /tmp/atlas-model-check.log 2>&1 || { cat /tmp/atlas-model-check.log; exit 1; }; \
      cat /tmp/atlas-model-check.log; \
      grep -E 'test result: ok. [1-9][0-9]* passed;' /tmp/atlas-model-check.log; \
    done
RUN LD_LIBRARY_PATH=/opt/cuda-stubs:/usr/local/cuda/lib64 \
    cargo test --locked --release -j "$BUILD_JOBS" -p spark-server --bin spark \
    scheduler::lifecycle_tests > /tmp/atlas-lifecycle-check.log 2>&1 || \
    { cat /tmp/atlas-lifecycle-check.log; exit 1; }; \
    cat /tmp/atlas-lifecycle-check.log && grep -E 'test result: ok. [1-9][0-9]* passed;' /tmp/atlas-lifecycle-check.log

RUN LD_LIBRARY_PATH=/opt/cuda-stubs:/usr/local/cuda/lib64 \
    cargo test --locked --release -j "$BUILD_JOBS" -p spark-server --bin spark \
    auto_tool_eos_tests > /tmp/atlas-tool-stop-check.log 2>&1 || \
    { cat /tmp/atlas-tool-stop-check.log; exit 1; }; \
    cat /tmp/atlas-tool-stop-check.log && grep -F 'test result: ok. 7 passed;' /tmp/atlas-tool-stop-check.log

FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04@sha256:95318efecfd68ab3d109da5277863257b06137c84f34a87f38de970d5cd035d3
ARG SPARKGLM_REVISION
RUN test -n "$SPARKGLM_REVISION"
LABEL org.opencontainers.image.licenses="AGPL-3.0-only" \
      org.opencontainers.image.source="https://github.com/Enntity/sparkglm" \
      org.opencontainers.image.revision="$SPARKGLM_REVISION"
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends --allow-change-held-packages \
    python3 ca-certificates ffmpeg libnccl2=2.31.2-1+cuda13.3 libibverbs1 librdmacm1 \
    ibverbs-providers libnl-3-200 libnl-route-3-200 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/target/release/spark /usr/local/bin/spark
COPY --from=builder /build/jinja-templates/ /jinja-templates/
COPY --from=native /opt/atlas/native/*.so /opt/atlas/native/
COPY --from=native /opt/atlas/native/native-libraries.sha256 /opt/atlas/native/
COPY --from=native /opt/atlas/native/notices/ /opt/atlas/notices/
COPY --from=builder /opt/cutlass/LICENSE.txt /opt/atlas/notices/CUTLASS-LICENSE.txt
COPY --from=native /tmp/FlashKDA-atlas/cutlass/LICENSE.txt /opt/atlas/notices/FlashKDA-CUTLASS-LICENSE.txt
COPY --from=native /product/research/atlas/nvidia-mtp-converter/ /opt/atlas/converter/
COPY product/research/atlas/install/serve.py product/research/atlas/install/profile.json /opt/atlas/
COPY source-manifest.json /opt/atlas/source-manifest.json
COPY engine/LICENSE /LICENSE
ENV LD_LIBRARY_PATH=/opt/atlas/native:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64
ENV RUST_LOG=info
EXPOSE 8893
ENTRYPOINT ["python3", "/opt/atlas/serve.py"]
