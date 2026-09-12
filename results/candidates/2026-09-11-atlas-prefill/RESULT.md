# Atlas NVIDIA prefill and MTP research

Research remains unpromoted. The final r11b canary passes four short cases and
the first field-guide prompt:15,807 input tokens,8 output tokens,13.087591 seconds
client TTFT. The recorded vLLM reference is10.848938 seconds. Atlas has not
reached parity; this is not a completed C4 benchmark.

The r10 field-guide result was13.067441 seconds. R11b reduces the observed
client-minus-server gap from358.857 to33.577 milliseconds, but its total client
TTFT is neutral. The sanitizer fix is not presented as an endpoint speedup.

The selected configuration retains native MTP with two drafts,32,768 context,
four sequence slots and4,096-token prefill chunks. Both ranks loaded the final
binary with11,189/10,336 target KV blocks and8,196 predictor blocks per rank.
The first0.91-utilization load correctly refused7,954 blocks below the8,196
floor; the0.914 retry passed. No OOM occurred in these attempts.

Earlier short-context MTP1 experiments showed modest coding gains and preserved
the tested outputs. They do not establish MTP2 long-context throughput or broad
quality. Predictor experts were quantized offline with Atlas's native runtime
quantizer, which differs from the reference predictor's BF16 execution.

Rejected standalone alternatives remain disabled. Transient-SFB CUTLASS and its
cooperative K256 alternative passed the operand/output oracles but were slower
than the selected M64 kernel, including required preparation. No production
integration was made. The source patches retain rejected experiments as such.

`measurement.json` separates the bounded model observations, earlier coding
runs and final capacity. `operator-evidence.json` retains selected numerical and
timing lines with original-source hashes. `validation.json` records focused
tests; `source.json` binds the reconstruction and executed binary. Full machine
logs, model metadata, weights, libraries, credentials and host locators are not
distributed. Complete G1–G5 qualification remains outstanding.
