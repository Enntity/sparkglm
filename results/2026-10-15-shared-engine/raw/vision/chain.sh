#!/bin/bash
# chain.sh : wait for both CI runs, then build the images (vis_build.sh) and run the tests and RigMark (vis_run.sh).
until grep -q CI-EXIT ~/ci-as-glmV.log && grep -q CI-EXIT ~/ci-as-qwenV.log; do sleep 20; done
echo "##### CI finished $(date -Is)"; sleep 30
ATLAS_BUILD_JOBS=8 bash vis_build.sh > vis-build.out 2>&1 || { echo BUILD-FAILED; tail -20 vis-build.out; exit 1; }
tail -4 vis-build.out; sleep 90
bash vis_run.sh > vis-run.out 2>&1
echo "##### chain done $(date -Is)"
