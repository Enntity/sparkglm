# SPDX-License-Identifier: MIT
# Adapted from loud1990 at d00b2ffa70e0ccbfd582088ab32448b01bfe9c67; pin base and include rank entrypoints.
FROM ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:9bb1557a4234fce63d59599e44d10747eabd742beb337eebf9e7070be8a0fd58

COPY overlay/qwen3_dflash2.py /opt/glm53/qwen3_dflash2.py
COPY overlay/patch_dflash2.py /opt/glm53/patch_dflash2.py

RUN python3 /opt/glm53/patch_dflash2.py \
    && python3 -m py_compile \
        /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dflash.py \
        /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dflash2.py \
    && grep -q 'self._fused_kv_weight_scale' \
        /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dflash.py \
    && grep -q 'quant_config=self.quant_config' \
        /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dflash2.py

COPY head.sh worker.sh /opt/sparkglm-nvfp4/
LABEL ai.enntity.sparkglm.experiment="nvfp4-tp2-unqualified"
