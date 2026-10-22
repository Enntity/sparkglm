#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Run as root on each Spark: cap the GB10 GPU clock and keep the cap across
# reboots. `./start.sh clock-cap` runs it on both Sparks.
#
#   gpu-clock-cap.sh [MAX_MHZ]    default 2200; the lock is 300,MAX_MHZ
#
# Under sustained full GPU load (a long cold prefill), an uncapped Spark can
# power itself off with nothing in any log. The cap prevents it; see
# results/2026-10-13-clock-cap/.
set -euo pipefail
max=${1:-2200}
[[ $max =~ ^[0-9]+$ ]] || { echo "MAX_MHZ must be a number of MHz" >&2; exit 2; }

cat > /etc/systemd/system/gpu-clock-cap.service <<EOF
[Unit]
Description=Cap GB10 GPU clocks at 300,$max MHz (SparkGLM)
After=nvidia-persistenced.service

[Service]
Type=oneshot
ExecStart=/usr/bin/nvidia-smi -lgc 300,$max
ExecStop=/usr/bin/nvidia-smi -rgc
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable gpu-clock-cap.service >/dev/null
# restart, not start: re-applies the lock when the unit already runs with another cap
systemctl restart gpu-clock-cap.service
echo "$(hostname): gpu-clock-cap $(systemctl is-active gpu-clock-cap.service), cap 300,$max MHz"
