# SPDX-License-Identifier: MIT AND Apache-2.0
# Add the pinned Mia E3 module; preserve the complete SparkGLM reference.
ARG BASE_IMAGE=sparkglm-release:video-clean
FROM ${BASE_IMAGE}
COPY research/experiments/exl3-e3 /opt/glm53/e3
ENV MAX_JOBS=2
RUN python3 /opt/glm53/e3/build_exl3_fat_moe_ext.py \
    --src /opt/glm53/e3 --out /opt/sparkglm-e3-build \
    --install /usr/local/lib/python3.12/dist-packages
ARG SPARKGLM_SOURCE_REVISION=unknown
LABEL org.enntity.sparkglm.build-profile="candidate-e3" \
      org.opencontainers.image.revision=${SPARKGLM_SOURCE_REVISION}
