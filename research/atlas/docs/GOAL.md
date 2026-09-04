# SparkGLM Atlas goal

Build a correctness-first, two-DGX-Spark serving path for the official
`zai-org/GLM-5.3-Flash` checkpoint on top of Atlas.

The finished system must run the checkpoint's exact 34-layer KDA and 11-layer
sparse NoPE-MLA topology, 4-stream mHC, 288-expert top-8 MoE, multi-EOS chat and
tool protocol, and optional MTP without substituting Qwen GDN or ordinary MLA
semantics. It must shard experts across two GB10 systems before materializing
rank-local weights, batch/interleave prefill and decode through a collective-
safe EP protocol, and admit requests from measured KV, KDA, indexer, scratch,
and weight budgets.

Success means:

- token and logit parity against an authoritative reference on short golden
  prompts before optimization;
- stable OpenAI-compatible text and tool serving at EP=2/TP=1;
- increasing concurrency improves aggregate throughput without unbounded TTFT;
- cancellation, preemption, prefix reuse, and speculation are enabled only
  after atomic KDA/indexer state snapshot and rollback tests pass;
- every performance claim is backed by a reproducible two-Spark measurement.

Current status: the full EXL3 checkpoint loads and serves on two Sparks, with
text, native tool-call, cancellation/slot-reuse, graph replay, and bounded C4
canaries passing. The finish line still requires authoritative model parity,
C8/long/endurance/failure testing, byte-aware admission beyond four fixed
slots, and atomic state support before MTP or prefix reuse.
