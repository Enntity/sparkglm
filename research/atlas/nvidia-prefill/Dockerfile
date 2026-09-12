# SPDX-License-Identifier: AGPL-3.0-only
# Adapted from Mango-kid/atlas 90b3584abc71b44b609637092b85d8423d8ff20f:
# docker/gb10/glm-5.3-flash-nvfp4/nvfp4/Dockerfile and docker/gb10/Dockerfile.
# Source-reviewed resume recipe; a fresh build of this file is not yet tested.
# Context must be the reconstructed Atlas tree. Build natively on DGX Spark.
FROM nvidia/cuda:13.0.0-devel-ubuntu24.04@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d AS builder

RUN apt-get update -qq && apt-get install -y -qq --allow-change-held-packages \
    curl build-essential pkg-config git cmake libclang-dev libibverbs-dev librdmacm-dev \
    libnccl2=2.31.2-1+cuda13.3 libnccl-dev=2.31.2-1+cuda13.3 && \
    rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --profile minimal --default-toolchain 1.93.1
ENV PATH="/root/.cargo/bin:${PATH}"
ENV RUSTUP_TOOLCHAIN=1.93.1
ENV CUDA_HOME=/usr/local/cuda
ENV CUTLASS_HOME=/opt/cutlass
RUN git clone --filter=blob:none https://github.com/NVIDIA/cutlass.git /opt/cutlass && \
    git -C /opt/cutlass checkout cf064d2e6bad2886238ac565b3b49007764f4939
WORKDIR /build
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates/ crates/
COPY kernels/ kernels/
COPY vendor/ vendor/
COPY jinja-templates/ jinja-templates/
COPY tests/fixtures/ tests/fixtures/
ENV ATLAS_TARGET_HW=gb10
ENV ATLAS_TARGET_MODEL=glm-5.3-flash-nvfp4
ENV ATLAS_TARGET_QUANT=nvfp4
RUN cargo build --locked --release -p spark-server

FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04@sha256:95318efecfd68ab3d109da5277863257b06137c84f34a87f38de970d5cd035d3
LABEL org.opencontainers.image.licenses="AGPL-3.0-only"
ARG ATLAS_SOURCE_REVISION=dd0ffd157fb7c6e96d2f3858c888513c6f142cf6
LABEL org.opencontainers.image.revision="${ATLAS_SOURCE_REVISION}"
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends --allow-change-held-packages \
    libnccl2=2.31.2-1+cuda13.3 libibverbs1 librdmacm1 ibverbs-providers \
    libnl-3-200 libnl-route-3-200 && rm -rf /var/lib/apt/lists/*
COPY --from=builder /build/target/release/spark /usr/local/bin/spark
COPY --from=builder /build/jinja-templates/ /jinja-templates/
COPY LICENSE /LICENSE
ENV RUST_LOG=info
EXPOSE 8893
ENTRYPOINT ["spark"]
