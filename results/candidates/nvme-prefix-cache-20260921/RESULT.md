# SSD prefix-cache feasibility: bounded TP2 measurement

The focused 200,000-token test restored 197,120 tokens from SSD and produced
the same continuation as cold and hot execution. The feature remains opt-in;
this bundle is not a standard G3 qualification or a promoted default.

| Mode | First visible token | Cached prompt tokens |
| --- | ---: | ---: |
| Cold | 132.477 s | 0 |
| Hot GPU cache | 2.408 s | 197,120 |
| GPU cleared, SSD restore | 5.640 s | 197,120 |
| Fresh serving process, retained SSD | 12.528 s | 197,120 |

The restore incurred 1,593,094,144 physical read bytes on rank zero. Both TP
ranks used their own local SSD. After the managed serving restart, the first
request reused the same 197,120-token prefix, returned the same continuation,
and incurred 1,786,957,824 physical read bytes on rank zero. Rank-zero payload storage after the cold
request was about 7.3 GiB, including intermediate hybrid-state checkpoints;
restore needs only the relevant prefix pages and recurrent-state boundary.

Four arrivals spaced 0.5 seconds apart after another GPU-cache reset returned
identical continuations at 6.936, 7.993, 9.750 and 9.733 seconds. They share one
200K prefix; this does not demonstrate four independent resident 200K sessions.
Both full-model probes ended with zero preemptions, no running requests and
no queued requests. Per-request IO counters overlap in the concurrent run and
must not be summed.

## Correctness evidence

- Sixteen focused local tests passed: short IO, corrupt/short/wrong-size
  payloads, commit publication, quota/reserve and namespace behavior.
- The CUDA/SSD byte roundtrip preserved all 385,152 bytes of heterogeneous pages.
- An 8K tinyGLM TP2 probe produced the same 32 output tokens cold, hot and
  SSD-restored. SSD restore took 0.104 s and incurred 8,937,472 physical read bytes.
- Removing the tiny test's rank-one payloads forced cold recomputation without
  changing the output or killing the server.
- The actual NVFP4 model retained adaptive DFlash and CUDA graphs. All observed
  full-model continuations returned the same seven output tokens.

## Scope and limits

This follows the requested minimal-iteration budget: one retained sample per
mode, rather than a full performance matrix. Cold execution includes first-use
JIT work. The large resume-latency difference supports this specific use case;
it is not an isolated kernel speedup, a broad quality evaluation, or endurance
certification. The standard G2/G3/G4 matrices and G5 endurance were not run.

The public default remains disabled. Storage is capped at 64 GiB per rank and
namespace with 64 GiB of free-space reserve. There is no online TTL/LRU cleanup:
new stores stop at the limit, and offline cleanup requires stopped ranks.

The private managed recipe now enables this option while retaining adaptive
DFlash. The normal server has dev mode disabled. Local model and alias
canaries passed after the restart. Existing sessions need one prefill on this
build before their prefix is available on SSD.
