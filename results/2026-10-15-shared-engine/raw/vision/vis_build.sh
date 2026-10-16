#!/bin/bash
# vis_build.sh : build both release images (vision engine) from their public pins (release/shared-engine-20261015)
# and load them on 02. No server runs; vis_run.sh tests them.
set -u; cd ~/vis-ab; W=RANK1_HOST; B=release/shared-engine-20261015
echo "##### start $(date -Is)"
for r in sparkglm sparkqwen; do
  rm -rf ~/vis-$r; git clone -q -b $B ~/$r-shared-20261014.git ~/vis-$r || exit 1
  ssh -n $W "rm -rf ~/vis-$r; git clone -q -b $B ~/$r-shared-20261014.git ~/vis-$r" || exit 1
  echo "== $r $(git -C ~/vis-$r log --oneline -1) install $(git -C ~/vis-$r rev-parse HEAD:install | cut -c1-12)"
done
GI=$(cd ~/vis-sparkglm && bash install/build.sh --tag); QI=$(cd ~/vis-sparkqwen && bash install/build.sh --tag)
[ "$GI" = ghcr.io/enntity/atlas-sparkglm:f0f8a55c3da2 ] && [ "$QI" = ghcr.io/enntity/atlas-sparkqwen:4e29c8c22671 ] || { echo "unexpected tags $GI $QI"; exit 1; }
for r in sparkglm sparkqwen; do
  echo "== build $r $(date +%T)"; (cd ~/vis-$r && bash install/build.sh) > build-$r.log 2>&1 || { echo "build $r FAILED"; tail -40 build-$r.log; exit 1; }
  tail -3 build-$r.log; done
echo "== load on 02 $(date +%T)"; docker save $GI $QI | ssh $W docker load || exit 1
for i in $GI $QI; do echo "$i 01 $(docker image inspect -f {{.Id}} $i | cut -c8-19) 02 $(ssh -n $W docker image inspect -f {{.Id}} $i | cut -c8-19)"; done
echo "##### built $(date -Is)"
