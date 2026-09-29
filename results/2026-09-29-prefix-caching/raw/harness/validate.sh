#!/bin/bash
# Build the fix branch from a fresh clone, start it, and run the full validation.
set -e
cd ~/sparkglm-pc/sparkglm && ./start.sh stop >/dev/null 2>&1 || true
cd ~/sparkglm-pc && rm -rf sparkglm && git clone -q -b fix/prefix-caching https://github.com/Enntity/sparkglm.git && cd sparkglm
printf 'WORKER=192.0.2.2\nMODEL_ROOT=%s\nBUILD=1\n' "$HOME/sparkglm-cleanroom/models" > .env
echo "== $(git log --oneline -1) $(install/build.sh --tag)"
./start.sh 2>&1 | grep -E "^.{0,12}== |smoke" | sed "s/\x1b\[[0-9;]*m//g"
set +e
docker logs atlas-sparkglm-rank0 2>&1 | sed "s/\x1b\[[0-9;]*m//g" | grep -E "Prefix caching|Marconi [0-9]+ slots|max KV tokens" | cut -c30-250
cd ~/sparkglm-dev
echo "== pc_test"; python3 pc_test.py v2 2>&1 | tail -9
echo "== pc_hard"; python3 pc_hard.py v2 2>&1 | grep -vE "^\s|Traceback|File " | tail -26
echo "== acc 1"; python3 pc_acc.py v2a 2>&1 | tail -1
echo "== acc 2"; python3 pc_acc.py v2b 2>&1 | tail -1
OUT=~/sparkglm-pc/out; mkdir -p $OUT; cd ~/sparkglm-pc/sparkglm/bench
echo "== matrix"; MATRIX_SALT=mb-vx2-7c31 python3 run_matrix.py $OUT/out-matrix-1 matrix
echo "== field"; MATRIX_SALT=fg-$(date +%s) python3 run_matrix.py $OUT/out-field field
docker logs atlas-sparkglm-rank0 2>&1 | sed "s/\x1b\[[0-9;]*m//g" | grep -E "Marconi|Prefix cache hit|shortcut" | cut -c30-230 > ~/sparkglm-dev/v2.snaps.log
echo VALIDATE-DONE
