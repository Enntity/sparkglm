# GLM-5.3-Flash two-Spark live validation — 2026-08-31

> Historical EP=2 bootstrap receipt. The optimized appliance is now pure TP=2
> with DFlash2 and cross-sequence target/draft batching; see
> [`NATIVE_TP2_VALIDATION_2026-09-01.md`](NATIVE_TP2_VALIDATION_2026-09-01.md).

These receipts are from the Atlas `glm53-flash` branch serving
`Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` with EP=2/TP=1 on two DGX Sparks.
They are measured endpoint behavior, not an authoritative model-parity proof.

## Correctness and recovery receipts

- Both ranks ran the same release binary, SHA-256
  `4b74c895535b55af29341d51058aa37fe64338f4c4026fdc1d4b0130d0d78040`.
- Thinking-disabled text returned exact `SPARK_OK` with zero reasoning tokens.
- A forced `get_weather` call returned OpenAI arguments
  `{"city":"Phoenix"}`, `finish_reason=tool_calls`, and zero reasoning tokens.
- Aborting a live stream after 1.002 seconds did not poison its slot: the
  immediate next request returned exact `SLOT_OK` at 534.64 ms TTFT.
- Health remained ready after the C4 and cancellation probes, with no observed
  CUDA illegal-address or NCCL asynchronous faults.

## Controlled prefill/decode overlap

The fixture requested about 256 prompt tokens but tokenized to 498. Request B
started eight seconds after A; each was forced to generate at least 64 tokens.

| Active-decode prefill slab | A max token gap | A effective decode | B TTFT | Total wall |
|---:|---:|---:|---:|---:|
| 128 tokens | 2.028 s | 2.83 tok/s | 9.32 s | 32.51 s |
| 32 tokens | 0.843 s | 4.32 tok/s | 15.90 s | 32.38 s |

The 32-token slab cut A's worst visible freeze by 58% and raised A's effective
stream rate by 53%, while increasing B's TTFT by 71%. Total wall time was
effectively unchanged. This is a latency-allocation tradeoff, not free
throughput or simultaneous kernel execution.

## Four-request canary

With 500 ms arrivals, 498-token prompts, 30 output tokens, a 128-token normal
prefill limit, and a 32-token active-decode slab, all four requests completed
in FIFO order:

| Request | TTFT | Effective decode | Max token gap |
|---:|---:|---:|---:|
| 0 | 7.41 s | 4.55 tok/s | 0.735 s |
| 1 | 18.33 s | 4.57 tok/s | 0.729 s |
| 2 | 29.09 s | 4.41 tok/s | 0.746 s |
| 3 | 40.21 s | 8.36 tok/s | 0.136 s |

Wall time was 45.06 seconds. This proves bounded streaming and correct FIFO
promotion at C4. It also exposes the remaining limitation: prompt completion
is still substantially serialized, so later-request TTFT grows with queue
depth. A future collective-safe mixed EP batch is the main aggregate-
throughput opportunity.

## Reproduce

```bash
python3 bench/glm53/staggered_openai.py \
  --base-url http://127.0.0.1:8888 \
  --model GLM-5.3-Flash-EXL3 \
  --concurrency 2 --stagger-ms 8000 \
  --prompt-tokens 256 --output-tokens 64 --min-output-tokens 64
```

Record `max_inter_token_gap_s` as well as TTFT and average decode rate. Average
tokens per second alone hides the user-visible stalls this scheduler change is
designed to bound.
