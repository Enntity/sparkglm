# Native NVFP4 full-model screen

One exploratory repetition, not a promoted configuration. All frozen requests completed with 400 output tokens and correct request isolation.

| Case | Wall seconds | Median TTFT seconds |
| --- | ---: | ---: |
| c1-16k | 17.171 | 8.186 |
| c1-32k | 50.957 | 38.162 |
| c2-16k | 47.279 | 21.441 |
| c2-32k | 53.448 | 30.907 |
| c4-16k | 78.019 | 37.619 |

The original 262144-context / 8 GiB profile loaded weights but failed cache admission (9.31 GiB required; estimated maximum 78848). Correcting context to 65536 allowed startup. The resulting 66164-token pool is tight for the C2/C4 matrix; increase cache modestly for the tuned comparison.

Bounded semantics passed 15/16, with the same 91x89 failure seen on the reference. Warmup completed 20/20; its sampler-cache postcondition was explicitly skipped. The LLooM streamed tool canary completed in 1.850 s and gateway metrics attribute it to sparkglm-nvfp4.

The server manifest source revision identifies the benchmark checkout. The image is the preserved reference at 672df9e4155fa1fa12f6dc63dbd41f0fa7272ab7; both rank image IDs were independently verified.

The separately prepared Humming alternative currently falls back from FP4 to FP8 activations in its installed input schema. Treat that as a distinct compute path even though it reads the NVFP4 checkpoint. It is not native W4A4 proof.
