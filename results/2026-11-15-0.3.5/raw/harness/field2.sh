#!/bin/bash
# field2.sh TAG: cold staggered four-stream captures at 16K and 32K prompts on the engine now serving, then the matrix.
# The salt goes in the prompt (--prompt-salt): the engine does not key its prefix cache on cache_salt, so
# run_matrix.py field measures a warm wave after its warmup. Same salts for every TAG.
set -u; cd ~/thinkoff; T=$1; O=~/thinkoff/field-1115/$T; B=~/thinkoff/sparkglm/bench
export MATRIX_BASE_URL=${FIELD_URL:-http://127.0.0.1:8893} MATRIX_MODEL=${FIELD_MODEL:-glm-5.3-flash-atlas}
for sz in 16384 32768; do k=$((sz/1024)); mkdir -p $O/cold-${k}k; echo "=== cold-${k}k $(date -u +%T)"
  for n in discarded-warmup field-guide-r1 field-guide-r2 field-guide-r3; do
    python3 $B/four_stream_video.py capture --base-url $MATRIX_BASE_URL --model $MATRIX_MODEL --output $O/cold-${k}k/$n.json \
      --streams 4 --max-tokens 400 --prompt-tokens $sz --stagger-ms 1000 --prompt-style field-guide \
      --prompt-salt fc${k}-1115-$n --recipe-label $T --timeout-s 900 > /dev/null 2>$O/cold-${k}k/$n.err
    python3 -c "
import json,sys
try:
    d=json.load(open(sys.argv[1])); r=d[\"requests\"]
    print(json.dumps({\"case\":sys.argv[2],\"wall_s\":round(d[\"wall_s\"],2),\"tokens\":d[\"total_completion_tokens\"],\"prompt\":[q[\"prompt_tokens\"] for q in r],\"first\":[round(q[\"first_token_s\"],1) if q[\"first_token_s\"] else None for q in r],\"ended\":[round(q[\"ended_s\"],1) for q in r],\"err\":[q[\"error\"] for q in r if q[\"error\"]]}))
except Exception as e: print(sys.argv[2],\"FAILED\",e)
" $O/cold-${k}k/$n.json $n
  done
done
echo "=== matrix $(date -u +%T)"
for p in $(seq ${FIELD_PASSES:-3}); do MATRIX_SALT=mb-1115-p$p python3 $B/run_matrix.py $O/matrix-$p matrix; done
echo "=== FIELD2 END $T $(date -u +%T)"
