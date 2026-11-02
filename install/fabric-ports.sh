#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Prints this host's active addressed RDMA ports: device, interface, CIDR. Inventory
# is read on each host, with no device-name or physical-port convention.
for dev in /sys/class/infiniband/*; do
  [[ -f $dev/ports/1/state ]] || continue
  grep -qs ACTIVE "$dev/ports/1/state" || continue
  for net in "$dev"/device/net/*; do
    [[ -e $net ]] || continue
    while read -r cidr; do
      [[ -n $cidr ]] && printf '%s %s %s\n' "${dev##*/}" "${net##*/}" "$cidr"
    done < <(ip -4 -o addr show dev "${net##*/}" | awk '{print $4}')
  done
done
exit 0
