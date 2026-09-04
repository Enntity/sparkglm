# Clean release C4 replay

This is the first clean-source replay of the default posted-video profile at
repository commit `b3d0bbd61a2331fa3d6e9e3a09c2a0e6f375e69e`. It is a
three-run measurement, not a paired optimization qualification and not a G3
or G5 release certification.

## Result

After the full startup shape sweep and the fatal four-resident-stream capacity
gate, the same running server completed three independently salted repetitions:

| repetition | wall (s) | aggregate delivered output (tok/s) |
| --- | ---: | ---: |
| r1 | 89.492 | 23.514 |
| r2 | 88.264 | 23.786 |
| r3 | 90.692 | 23.038 |
| median | 89.492 | 23.514 |

Median TTFT by arrival lane was 21.414, 30.723, 39.146, and 45.172 seconds.
All twelve requests completed with 400 delivered tokens. The actual prompt
counts were 15,806-15,809 tokens, so this remains a 16K-class legacy-video
workload rather than the exact-token frozen G3 matrix.

The retained posted capture was one warmed sample at 86.149 seconds and 24.267
tok/s. The clean-build median is 3.1% lower in aggregate throughput and 3.9%
higher in wall time. That difference is inside the variation seen when the
historical image was replayed on this appliance, but this bundle does not claim
statistical equivalence. It establishes that the default clean build is close,
stable across three repetitions, and not suffering a large missing-path
regression.

## Identity and startup

- image on both ranks:
  `sha256:5dfeec28e67740b087a3f432505f8726784d703d813ecbc33b39587ad58822d4`;
- target revision:
  `25a44fdbf16862a46b7cc9921142c6c81350af2f`;
- DFlash2 revision:
  `7d74cdd881ed7e32c31175984a67823127b66cfe`;
- topology: TP=2 on two GB10 systems;
- runtime profile: mixed scheduling, grouped prefill, cooperative decode up to
  16 tokens, 7,168 batched tokens, four sequence slots, and FP8 target KV;
- startup passed the embedded image checks, GPU EXL3 self-check on both ranks,
  graph/shape warmup, and the four-resident-stream long C4 capacity gate.

`raw/server-manifest.json` reports a dirty checkout because the result bundle
was being created when the live manifest was captured. The running image was
built earlier from the clean `b3d0bbd` commit, and its immutable digest and
source label are the identities used here.

## Historical-image parity boundary

The clean image is a source-audited successor with the posted runtime profile;
it is not a byte-for-byte reconstruction of
`sparkglm-vllm:exl3-grouped-prefill-candidate-20260903d`. Of 2,423 installed
vLLM Python files, 10 differ. The compiled EXL3 extension also differs because
the clean repository rebuilds it from the published, attributed source chain.
`raw/source-parity.json` records the exact file list and hashes.

The substantive historical-only Python branches are native-FP16 sparse-indexer
score plumbing, optional decode-floor/phase-interleave scheduler experiments,
per-group prefix-cache retention, an alignment-floor fix, and FA4 discovery.
The posted runtime selected mixed scheduling (`GLM53_MIXED_PREFILL_CHUNK=0`),
so the optional decode-floor and phase-interleave policies were inactive. The
salted workload also does not test reusable-prefix behavior. The FP16 score
path and alignment-floor code are real implementation differences, so they are
not described as identical or silently credited to the clean build.

The proven current-best EXL3 features are present in the clean image: the M64
paired/fused expert path, GPU-resident grouped prefill, and cooperative decode.
The close three-run endpoint result is stronger evidence for the reconstructed
default than source-shape inference alone, while still falling short of the
full frozen G3/G4/G5 policy.

## Limitations

- No paired baseline/candidate run order was collected; this is a baseline
  reconstruction measurement.
- This is only the four-stream 16K-class workload, not the complete G3 matrix.
- No semantic-quality suite was run, so it is not G4 evidence.
- Speculative acceptance counters were captured for two repetitions in the
  operator journal but are not embedded in the client receipts.
- Thermals, clocks, and system background activity were not recorded.
