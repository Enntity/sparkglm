# Packed RMSNorm candidate — 2026-09-01

Candidate image `sparkglm-vllm:d3860af` contains the packed-output change from
vLLM PR #53109. The image label is `org.opencontainers.image.revision=d3860af`;
the launch recipe was commit `f7c2930`, which uses spawned CUDA workers to make
a cold start reliable. The focused GB10 CUDA probe changed output strides from
the inherited fused-projection `(768, 1)` to packed `(192, 1)` and `(576, 1)`.

## Real workload gate

This is the same medium/large C1/C2 matrix, five-second C2 stagger, 64 forced
output tokens, and exact prompt salt `2026-09-01-throughput-a` used by the
`e3f690c` control receipt. The candidate engine was restarted with an empty
prefix cache before these prompt bytes were reused.

| Case | TTFT candidate (s) | Effective prefill candidate (tok/s) | Mixed delivery candidate (tok/s) | Wall candidate (s) | Change vs control |
| --- | --- | --- | ---: | ---: | --- |
| Medium C1 | 15.67 | 1008.7 | 26.06 | 18.09 | prefill +9.7%, TTFT -8.8%, wall -7.3% |
| Medium C2 | 17.62 / 28.75 | 897.1 / 549.7 | 6.87 | 35.97 | prefill +9.2% / +8.2%, TTFT -8.4% / -7.6%, wall -4.8% |
| Large C1 | 31.44 | 1023.7 | 30.10 | 33.53 | prefill +2.7%, TTFT -2.6%, wall -1.3% |
| Large C2 | 35.27 / 60.17 | 912.4 / 534.7 | 3.94 | 67.26 | prefill +1.5% / +1.1%, TTFT -1.5% / -1.1%, wall -0.8% |

All six requests completed all 64 requested output tokens. Maximum visible
token gaps were 0.129–0.141 seconds, versus 0.135–0.154 seconds in the control.
Output hashes changed because the packed input selects a different downstream
GEMM path; the prompts use greedy decoding, so small numeric changes can alter
later tokens. For that reason, short per-request decode rates from these prose
outputs are retained but are not used alone as the kernel acceptance gate.

## Fixed structured decode probe

The candidate produced the same 400-token counting output at C1 and C2
(`0436adfa…`, completion reason `length`):

| Concurrency | Aggregate decode | Per-request decode | Prior steady baseline | Change |
| --- | ---: | --- | ---: | ---: |
| C1 | 64.1 tok/s | 64.1 tok/s | 49.5 tok/s | +29.5% |
| C2 | 93.6 tok/s | 49.8 / 47.6 tok/s | 86.7 tok/s | +7.9% |

The API did not expose DFlash accepted/rejected counters in this run (both were
reported as zero), so future matrix receipts retain those fields explicitly but
must not infer a 0% acceptance rate from a missing backend counter.

Verdict: accept. The port improves the target medium C1/C2 workload materially,
does not regress the large C2 wall/TTFT gate, and improves the content-stable
decode probe.

## 2026-09-02 isolation revalidation

The original matrix retained hashes but did not fail on a missing request
isolation marker. After that gate was added, the unchanged `d3860af` image was
restarted and rerun with fresh prompt salts. All six medium/large C1/C2 requests
completed 64/64 tokens, contained their own marker, contained no peer marker,
and returned no error. Fresh TTFT was 15.62 seconds for medium C1,
17.96/29.01 for medium C2, 31.73 for large C1, and 35.53/61.18 for large C2.

An additional exact salted probe used to diagnose the later physical-index
reuse candidate also passed on `d3860af`: coherent content, own marker present,
no foreign marker, and 64/64 output tokens. This strengthens the acceptance
verdict with an explicit end-to-end correctness gate.
