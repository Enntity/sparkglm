# Lessons carried from the Atlas implementation

These are engineering constraints learned from the preserved native work. They
are not reasons to retain Atlas as the active serving engine.

## Baseline and proof

- Copying or launching the reference implementation proves deployment and the
  benchmark harness; it is not native parity and must never be reported as such.
- Compare the candidate with the installed control and with its immediately
  preceding commit. Record exact image/source identity on both ranks.
- Operator microbenchmarks are necessary but insufficient. A 13-18x selector
  improvement translated into only a few percent endpoint gain.
- Short 64-token runs can reverse the verdict from a stable 400-token run. The
  real 16K/32K C1-C2 matrix is the primary decision gate.
- Target-verified output hashes can remain identical while numerical changes
  reduce speculative acceptance. Output, acceptance, and state correctness are
  separate receipts.
- Same-binary feature toggles are the strongest A/B when they do not alter graph
  shape or startup state. Otherwise use adjacent isolated commits and identical
  cold/warm procedures.

## GLM execution

- Long-context prefill, not EXL3 quantization itself, caused the largest native
  deficit. The Atlas path was about 2.8x slower at medium C1 and about 4.9x at
  large C1, with superlinear prompt growth.
- The first-chunk profile identified sparse DSA/NoPE-MLA as the dominant native
  prefill stage. Exact radix selection and BF16 tensor-core score calculation
  stacked to a real improvement, but attention/gather/projection remained the
  larger interior cost.
- GLM state is heterogeneous. KDA recurrent and convolution state, sparse MLA
  pages/index metadata, draft state, and rollback state cannot be treated as an
  ordinary uniform KV cache.
- Speculative TP changes require rank-symmetric target context capture. A TP2
  DFlash experiment initially proposed from stale worker context; correcting
  ownership restored correctness but the added collectives still lost end to
  end.
- Shape specialization matters. An EXL3 gate/up kernel won at C1-C2 and lost at
  wider C4 until dispatch became shape-adaptive. A prefill MLA kernel was much
  slower when reused for a single decode row.

## GB10 and TP2

- Isolated NCCL improvements did not necessarily transfer into a captured graph
  containing many reductions. Measure the production graph before changing
  collective policy.
- The tested BF16 KDA projections were already close to the bandwidth implied by
  reading their weights; cuBLASLt heuristic searching found no easy replacement.
  Material gains require fewer bytes or fusion, with a quality proof.
- Kernel launch count, global scratch traffic, and layout conversion often
  mattered more than nominal arithmetic throughput.
- Preserve warmed compilation caches and exact artifacts on both Sparks. JIT,
  SSH setup, model reload, and rank drift can overwhelm iteration time and
  invalidate comparisons.

## Scheduler

- Low KV utilization does not mean the engine has useful scheduling capacity.
- The current `skip` policy protects decode from catastrophic mixed sparse-MLA
  prefill interference, but gives a later cold request unbounded TTFT.
- Do not simply enable simultaneous mixed forwards. First shorten and bound
  prefill kernels, then implement rank-symmetric decode/prefill phase
  interleaving with explicit latency budgets.
