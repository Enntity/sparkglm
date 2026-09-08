# Staggered C4 video: historical baseline, previous build, current NVFP4

Four field-guide prompts, approximately 15.8K actual input tokens each, arrivals at 0/1/2/3 seconds, 400 output tokens each. Current runs use the normal LLooM gateway, distinct cache namespaces, and the unchanged 512K context / 9 GiB cache per rank recipe. Historical captures are the exact inputs used for the last comparison video.

| Capture | Wall seconds | First request TTFT seconds |
|---|---:|---:|
| Mia EXL3 eb0469f baseline, archived Sep 3 | 95.212 | 24.017 |
| SparkGLM EXL3 4b23759+c805318, previous video | 86.149 | 20.379 |
| Current NVFP4, median of three Sep 8 runs | 83.869 | 10.910 |

Current run walls: 77.556, 86.849, 83.869 seconds. All twelve requests delivered 400 output tokens without API errors. The median is 11.9% shorter than the archived baseline and 2.6% shorter than the previous capture, but the current run spread exceeds the latter difference; that small total-wall improvement is not conclusive. First output improved more substantially. Current capture remained subject to preemption: the counter increased from 3 at the first in-run snapshot to 13 at completion. No claim of contention-free performance.

The 2x2 video puts baseline top left, previous build top right, current bottom left, and leaves bottom right blank. All three play at 1x elapsed time from request start; no prefill trimming. The current panel uses repetition 3, the median by total wall time. Each panel replays actual SSE text timing. Token progress is an estimate distributed across text deltas from exact final server usage.

The MP4 is stored outside the source repository, with its SHA-256 and format metadata in video.json. Reproduce by assembling comparison.json from raw/panel-0.json through raw/panel-2.json and running render.py with that directory. No kernel, runtime, route, or recommended-default changes were made for this capture.
