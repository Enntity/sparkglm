# SPDX-License-Identifier: AGPL-3.0-only
# Optional AGPL derivative. Build from repository root, before the NVMe layer.
ARG BASE
FROM ${BASE}
COPY research/adaptive-verification /opt/sparkglm/adaptive
COPY LICENSE /opt/sparkglm/adaptive/LICENSE.Apache-2.0
COPY LICENSES/MIT-LLooM.txt /opt/sparkglm/adaptive/LICENSE.MIT-LLooM
RUN python3 -S /opt/sparkglm/adaptive/install.py
# Profile invokes this entrypoint directly so a stale host launcher cannot shadow it.
