# Four-stream capacity fix — 2026-09-03

## Verdict

The late collapse in the first SparkGLM video was not an intrinsic scheduler or
kernel limit. The live appliance had retained a profiling override that fixed
the KV cache at about 10 GiB. A single 16K request consumed about 67% of that
hybrid KDA/MLA state pool; with four staggered requests, vLLM could keep only
two resident and queued the other two.

Removing the override produced a 19.87 GiB cache with capacity for 1,240,418
tokens. A bounded live probe then observed all four 15,757-token requests
resident together (`max_running=4`), with peak KV use of 71.73%.

The exact video workload subsequently completed in 88.22 seconds. That is
24.04 seconds sooner than the existing Mia capture and 43.32 seconds sooner
than the incorrectly configured SparkGLM capture.

## Corrected appliance configuration

- `MAX_MODEL_LEN=1000000`
- `GPU_MEM_UTIL=0.87`
- `MAX_NUM_SEQS=4`
- `MAX_NUM_BATCHED_TOKENS=7168`
- no manual `--kv-cache-memory-bytes` override
- DFlash2 k=7, draft TP2
- mixed-prefill policy `0`
- right-sized indexer workspace
- 16 ms worker spin-wait

The launcher now rejects a manual KV cache below 18 GiB when four sequence
slots are requested. Before printing the ready banner, it also runs a fatal
four-stream 16K residency probe. This turns the exact failure mode from a
silent throughput regression into a startup error.

## Exact replay against the existing Mia capture

Both traces use the same four prompts, prompt salt, tokenization, 0/1/2/3
second arrival offsets, 400 required output tokens, temperature zero, thinking
off, and EOS ignored. The fixed SparkGLM trace identifies recipe commit
`e7e35579b805`; the existing Mia trace identifies commit `eb0469f`.

| Metric | Broken SparkGLM | Fixed SparkGLM | Mia |
| --- | ---: | ---: | ---: |
| First-token clock, stream 1 | 27.83 s | **20.53 s** | 36.38 s |
| First-token clock, stream 2 | 36.17 s | **30.70 s** | 51.65 s |
| First-token clock, stream 3 | 75.95 s | **40.95 s** | 64.05 s |
| First-token clock, stream 4 | 100.88 s | **46.44 s** | 70.38 s |
| Full wall time | 131.54 s | **88.22 s** | 112.26 s |
| Full-window delivery | 12.16 tok/s | **18.14 tok/s** | 14.25 tok/s |
| Post-first-token aggregate | 15.39 tok/s | **23.58 tok/s** | 21.03 tok/s |

Relative to the broken SparkGLM run, the fix cuts wall time by 32.9%, raises
full-window delivery by 49.2%, and raises post-first-token aggregate throughput
by 53.2%. Relative to the existing Mia capture, fixed SparkGLM cuts wall time
by 21.4%, delivers 27.3% more tokens per full wall-clock second, and reports
12.1% higher post-first-token aggregate throughput in this one replay.

This is still a single-run visual proof, not a variance-qualified performance
claim. There is also an operational warmup difference: fixed SparkGLM now
warms and validates the long C4 shape before declaring itself ready, whereas
the retained Mia capture followed Mia's 20-request short-shape boot warmup.
That difference is intentional production behavior, but it must be disclosed
when publishing the comparison.

## Artifacts

- `2026-09-03-sparkglm-fixed-c4-video.json`
- `glm53-four-stream-16k-mia-eb0469f.json`
- `sparkglm-fixed-c4-4stream-16k.mp4` (SparkGLM only)
- `sparkglm-fixed-vs-mia-top-bottom.mp4` (SparkGLM top, Mia bottom)

The video is a replay of captured SSE timestamps. The renderer does not
synthesize token arrival timing.
