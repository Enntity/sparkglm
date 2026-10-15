#!/bin/bash
# rel_build.sh : build both release images from their public pins (release/shared-engine-20261015), load them on 02,
# No server runs; rel_smoke.sh smokes them.
set -u; cd ~/rel-ab; W=RANK1_HOST; B=release/shared-engine-20261015
QPROD=ghcr.io/enntity/atlas-sparkqwen@sha256:ac3a5c7da4841cacc810755da01b074b8216224409d7937a19dcfced569b72f0
down() { for n in 0 1; do docker rm -f atlas-sparkglm-rank$n >/dev/null 2>&1; ssh -n $W "docker rm -f atlas-sparkglm-rank$n >/dev/null 2>&1"; done; }
echo "##### start $(date -Is)"
for r in sparkglm sparkqwen; do
  rm -rf ~/rel-$r; git clone -q -b $B ~/$r-shared-20261014.git ~/rel-$r || exit 1
  ssh -n $W "rm -rf ~/rel-$r; git clone -q -b $B ~/$r-shared-20261014.git ~/rel-$r" || exit 1
  echo "== $r $(git -C ~/rel-$r log --oneline -1) install $(git -C ~/rel-$r rev-parse HEAD:install | cut -c1-12)"
done
GI=$(cd ~/rel-sparkglm && bash install/build.sh --tag); QI=$(cd ~/rel-sparkqwen && bash install/build.sh --tag)
[ "$GI" = ghcr.io/enntity/atlas-sparkglm:948aabe47fa4 ] && [ "$QI" = ghcr.io/enntity/atlas-sparkqwen:9a0df5f1474a ] || { echo "unexpected tags $GI $QI"; exit 1; }
for r in sparkglm sparkqwen; do
  echo "== build $r $(date +%T)"; (cd ~/rel-$r && bash install/build.sh) > build-$r.log 2>&1 || { echo "build $r FAILED"; tail -40 build-$r.log; exit 1; }
  tail -3 build-$r.log; done
echo "== load on 02 $(date +%T)"; docker save $GI $QI | ssh $W docker load || exit 1
for i in $GI $QI; do echo "$i 01 $(docker image inspect -f {{.Id}} $i | cut -c8-19) 02 $(ssh -n $W docker image inspect -f {{.Id}} $i | cut -c8-19)"; done
echo "##### built $(date -Is)"
