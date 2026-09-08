# Independently tuned TP2 comparison

Experimental appliance measurements. Lower wall time is better. This compares complete configurations with different quantization, draft, context capacity, and cache policies; it is not an isolated kernel A/B. All corresponding prompts have identical hashes and exact tokenizer counts, and every request completed 400 tokens with correct isolation.

| Case | reference | nvfp4-1k | exl3-7k | exl3-1k | e3-fixed | nv-marlin | nvfp4-2k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| c1-16k | 24.51s (n=3) | 23.28s (n=3) | 24.15s (n=1) | 26.97s (n=1) | 23.37s (n=3) | 28.54s (n=1) | 21.25s (n=3) |
| c1-32k | 35.61s (n=3) | 31.02s (n=3) | 35.39s (n=1) | 34.43s (n=1) | 34.62s (n=3) | 31.25s (n=1) | 29.24s (n=3) |
| c2-16k | 40.02s (n=3) | 38.79s (n=3) | 40.63s (n=1) | 49.41s (n=1) | 36.85s (n=3) | 37.70s (n=1) | 34.87s (n=3) |
| c2-32k | 60.90s (n=3) | 53.40s (n=3) | 61.62s (n=1) | 66.81s (n=1) | 55.91s (n=3) | 60.15s (n=1) | 53.48s (n=3) |
| c4-16k | 78.54s (n=3) | 63.79s (n=3) | 74.02s (n=1) | 85.96s (n=1) | 67.31s (n=3) | 63.45s (n=1) | 57.21s (n=3) |

Repetition counts are shown per cell; any one-repetition arm is a screen only. Arms were measured at different times. The reference is the preserved posted-video image with a BF16 draft and a much larger context limit. Exact settings are in configuration.json; raw receipts are checksum-bound here.

Quality remains a separate constraint. Consult the bounded case results below and each arm's full semantic receipts. No general model-quality, endurance, or default-promotion claim is made.

## Configuration key and selection

`reference` is the posted-video EXL3 engine with BF16 DFlash2. `e3-fixed` is corrected concurrent E3 with BF16 DFlash2, 32 expert temporary rows and 7168-token chunks. `exl3-7k` and `exl3-1k` are the separate EXL3/MXFP8 draft screens. `nvfp4-1k` and `nvfp4-2k` use native W4A4 with the MXFP8 draft; `nv-marlin` uses the same NVFP4 weights with BF16 activations (W4A16). All use actual target and draft TP2 with seven draft tokens.

The EXL3 finalist is concurrent E3. The overall NVFP4 finalist is native 2K: it is effectively tied with native 1K on the primary C2-32K median and faster on the other retained median cases. Small timing differences are not treated as established universal wins. Marlin passes the bounded semantic suite but does not win the primary performance screen.

The EXL3 finalist is configured for 1,000,000 context tokens with automatic KV allocation; the NVFP4 finalist uses 65,536 context tokens and 9 GiB KV per rank. The performance workloads test 16K and 32K prompts, not the full configured context limits. These capacity settings must remain visible when choosing a profile.

## Bounded semantics

| Arm | Passed | Failed cases |
| --- | ---: | --- |
| reference | 15/16 | arithmetic_91_89 |
| nvfp4-1k | 15/16 | arithmetic_91_89 |
| exl3-7k | 14/16 | arithmetic_123_47, arithmetic_91_89 |
| exl3-1k | 14/16 | arithmetic_123_47, arithmetic_91_89 |
| e3-fixed | 14/16 | arithmetic_123_47, arithmetic_91_89 |
| nv-marlin | 16/16 |  |
| nvfp4-2k | 15/16 | arithmetic_91_89 |

## Streaming and response latency

Median values below use each retained run. Worst event gap is the largest observed gap between response events across all requests; it includes contention and is not a pure decode kernel measurement.

| Arm | Case | Median TTFT seconds | Worst event gap seconds |
| --- | --- | ---: | ---: |
| reference | c1-16k | 11.056 | 0.232 |
| reference | c1-32k | 21.519 | 0.271 |
| reference | c2-16k | 19.950 | 3.858 |
| reference | c2-32k | 35.712 | 5.172 |
| reference | c4-16k | 33.737 | 6.164 |
| nvfp4-1k | c1-16k | 9.847 | 0.236 |
| nvfp4-1k | c1-32k | 18.651 | 0.240 |
| nvfp4-1k | c2-16k | 15.505 | 0.737 |
| nvfp4-1k | c2-32k | 30.665 | 0.716 |
| nvfp4-1k | c4-16k | 26.415 | 1.287 |
| exl3-7k | c1-16k | 12.021 | 0.167 |
| exl3-7k | c1-32k | 23.482 | 0.165 |
| exl3-7k | c2-16k | 21.706 | 2.976 |
| exl3-7k | c2-32k | 38.327 | 5.544 |
| exl3-7k | c4-16k | 35.849 | 6.044 |
| exl3-1k | c1-16k | 14.153 | 0.172 |
| exl3-1k | c1-32k | 25.884 | 0.246 |
| exl3-1k | c2-16k | 20.613 | 0.991 |
| exl3-1k | c2-32k | 39.409 | 0.849 |
| exl3-1k | c4-16k | 34.189 | 1.712 |
| e3-fixed | c1-16k | 10.949 | 0.379 |
| e3-fixed | c1-32k | 24.884 | 0.224 |
| e3-fixed | c2-16k | 19.072 | 3.091 |
| e3-fixed | c2-32k | 32.161 | 4.423 |
| e3-fixed | c4-16k | 29.638 | 4.739 |
| nv-marlin | c1-16k | 10.687 | 0.206 |
| nv-marlin | c1-32k | 20.228 | 0.173 |
| nv-marlin | c2-16k | 16.494 | 0.734 |
| nv-marlin | c2-32k | 32.931 | 0.746 |
| nv-marlin | c4-16k | 28.574 | 0.810 |
| nvfp4-2k | c1-16k | 8.911 | 0.149 |
| nvfp4-2k | c1-32k | 17.113 | 0.234 |
| nvfp4-2k | c2-16k | 14.067 | 1.137 |
| nvfp4-2k | c2-32k | 27.053 | 1.156 |
| nvfp4-2k | c4-16k | 23.273 | 1.260 |


## Sampled thermal state

Five-second GPU telemetry and approximate benchmark-window alignment (using original output timestamps, not copied bundle timestamps) are retained in thermal-summary.json and the raw CSVs. This distinguishes thermal slowdown from ordinary power capping. One software thermal sample occurred during the single-run EXL3/MXFP8 7K screen; treat that screen with this additional limitation. Coverage is incomplete: the first reference repetition predates the available telemetry, and renewal of the collector left a short gap during Marlin. Absence of a sampled flag does not rule out shorter events.
