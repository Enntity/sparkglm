#!/bin/bash
# cells.sh TAG [gbench extra args]: the comparison cells on whatever engine the extra args point at (default SparkGLM).
T=$1; shift; X=("$@"); cd ~/thinkoff; mkdir -p cells
{ J=${JSON_EFFORT:-off}; for spec in "1 text low" "2 text low" "4 text low" "1 json $J" "2 json $J" "4 json $J"; do set -- $spec
    for ctx in 60 2000; do
      python3 gbench.py --streams $1 --ctx $ctx --mode $2 --effort $3 --salt cells-$T --warm --out 64 "${X[@]}" > /dev/null 2>&1
      python3 gbench.py --streams $1 --ctx $ctx --mode $2 --effort $3 --salt cells-$T --warm --out 400 "${X[@]}" 2>&1 | tail -1 | python3 -c "
import sys,json
try:
    d=json.loads(sys.stdin.read()); print(d['case'].ljust(30), 'e2e', str(d['e2e_tok_s']).rjust(5), 'wall', str(d['wall_s']).rjust(5), 'decode-agg', str(d['aggregate_decode_tok_s']).rjust(5), 'per-stream', d['per_stream_tok_s'], 'ttft', d['ttft_s'], 'max_gap', d['max_gap_s'], 'reasoning', d['reasoning'])
except Exception as e: print('ERROR', e)"
    done
  done
  python3 gbench.py --streams 2 --ctx 30000 --mode text --effort low --salt cold-$T --warm --warm-n 1 --stagger 1 --out 600 "${X[@]}" 2>&1 | tail -1 | cut -c1-420
} | tee cells/$T.out
