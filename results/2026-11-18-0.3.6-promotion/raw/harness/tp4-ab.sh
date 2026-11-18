#!/bin/bash
# tp4-ab.sh TAG: start the four-Spark chain as configured and run the qualification matrix only (receipts in ~/sg-036/ab-TAG).
set -u; export PATH=$HOME/.local/bin:$PATH; Q=~/qual-20261103/kit; T=$1; D=~/sg-036/ab-$T; mkdir -p $D; cd $D
sleep 40
for try in 1 2 3 4 5; do out=$(lloom resume glm-5.3-flash-atlas --apply --yes 2>&1 | grep -iE "applied|error|cannot" | head -2); echo "$out" | cut -c1-200; echo "$out" | grep -qi "error\|cannot" || break; sleep 45; done
for i in $(seq 120); do curl -s -m 5 localhost:8893/v1/models | grep -q glm-5.3-flash-atlas && break; sleep 5; done
c=$(docker ps --format "{{.Names}}" | grep -m1 sparkglm); docker inspect $c --format "{{.Config.Image}}" | tee image
echo "=== serving $T $(date -u +%T)"
export SQ_MODEL=glm-5.3-flash-atlas
MATRIX_SALT=mb-vx2-7c31 python3 $Q/sg/run_matrix.py matrix matrix > matrix.log 2>&1; echo "matrix rc $?"
tail -5 matrix.log | cut -c1-200
lloom suspend glm-5.3-flash-atlas --apply --yes 2>&1 | grep -E "applied|Error" | head -1
echo "=== AB DONE $T $(date -u +%T)"
