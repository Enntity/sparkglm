#!/bin/bash
# Host-side guard for the gpustress container: kill it if any thermal zone reaches LIMIT C.
# Logs the hottest zone and the running peak every 10 s to /tmp/stressguard.log.
limit=${1:-98}
peak=0; n=0
while [ "$(docker inspect -f '{{.State.Status}}' gpustress 2>/dev/null)" = running ]; do
  t=$(( $(cat /sys/class/thermal/thermal_zone*/temp | sort -n | tail -1) / 1000 ))
  [ $t -gt $peak ] && peak=$t
  if [ $t -ge $limit ]; then docker kill gpustress >/dev/null; echo "$(date +%T) KILLED zone=${t}C peak=${peak}C"; break; fi
  n=$((n+1)); [ $((n % 10)) -eq 0 ] && echo "$(date +%T) zone=${t}C peak=${peak}C"
  sleep 1
done
echo "$(date +%T) END peak=${peak}C"
