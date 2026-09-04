# TP2 sparse-indexer row sharding — rejected — 2026-09-01

Candidate `36a42f1` ported the GLM-relevant core of vLLM PR #54394 on top of
accepted packed-RMSNorm image `d3860af`. It assigns complete prefill query rows
to one TP rank, then all-gathers each row's 2,048 int32 selected indices.

The metadata unit probe passed: a lopsided 4,096-row case was split 1,577 / 2,519,
covered every row once, balanced scored-key work within 2%, and declined below
the 2,048-row TP2 threshold. The real engine enabled the feature, completed
graph capture, served all boot canaries, and all matrix requests completed.

The exact `2026-09-01-throughput-a` prompt bytes were then run on an empty
prefix cache and compared with the accepted `d3860af` candidate:

| Case | Effective prefill | TTFT | Wall | Verdict |
| --- | --- | --- | ---: | --- |
| Medium C1 | 1008.7 → 946.6 tok/s (-6.2%) | 15.67 → 16.70 s (+6.6%) | 18.09 → 19.04 s (+5.2%) | regress |
| Medium C2 | 897.1/549.7 → 890.0/546.7 (-0.8%/-0.5%) | 17.62/28.75 → 17.76/28.91 s | 35.97 → 35.89 s | noise/parity |
| Large C1 | 1023.7 → 978.7 tok/s (-4.4%) | 31.44 → 32.88 s (+4.6%) | 33.53 → 34.88 s (+4.0%) | regress |
| Large C2 | 912.4/534.7 → 908.3/533.2 (-0.5%/-0.3%) | 35.27/60.17 → 35.43/60.34 s | 67.26 → 67.28 s | parity |

At C1 the all-gatherv moves roughly 16 MiB per full 2,048-row indexer step per
sparse layer, and that CX7 exchange costs more than the duplicate score/top-k
work it removes. In staggered C2, decode token reservations usually leave fewer
than 2,048 prefill rows, so the upstream threshold declines to shard and the
candidate stays near parity.

Verdict: reject and revert from the active branch. Preserve the experiment for
future work on a compressed exchange (smaller candidate set, hierarchical
selection, or communication/compute overlap); lowering the row threshold alone
would make the target mixed workload pay this losing exchange more often.
