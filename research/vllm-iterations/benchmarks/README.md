# SparkGLM real-workload gate

This directory carries the dependency-free endpoint harness developed during
the preserved Atlas implementation. The two Python files remain
AGPL-3.0-only; the adjacent `LICENSE` applies to them. They are benchmark tools,
not part of the vLLM serving process.

## Design

- **Purpose:** compare two OpenAI-compatible GLM endpoints under identical
  uncached medium/large C1-C2 traffic.
- **Input:** endpoint, served model name, optional API key, fixed case subset,
  five-second C2 stagger, output length, and a run-specific salt.
- **Output:** one JSON row per case plus a combined JSON object containing
  actual prompt/completion tokens, TTFT, effective prefill rate, active decode
  rate, aggregate appliance decode rate, maximum token gap, output hash,
  completion status, and wall time.
- **Failure guarded:** accidental prefix hits, incomplete generations, request
  cross-contamination, endpoint errors, and benchmark-shape drift.
- **Cheapest validation:** bytecode compilation and CLI/config parsing locally;
  full behavioral validation requires an OpenAI-compatible streaming endpoint.

The stable primary matrix is:

| Case | Approximate input | Requests | Arrival offsets |
| --- | ---: | ---: | --- |
| `medium-c1` | 16K | 1 | 0 s |
| `medium-c2` | 16K each | 2 | 0 s, 5 s |
| `large-c1` | 32K | 1 | 0 s |
| `large-c2` | 32K each | 2 | 0 s, 5 s |

Example:

```bash
.venv/bin/python benchmarks/sparkglm/real_workload_matrix.py \
  --base-url http://127.0.0.1:8888 \
  --model GLM-5.3-Flash-EXL3 \
  --tag candidate-c707598 \
  --salt-variant 2026-09-01-a
```

Use a new `--salt-variant` for every repeated comparison. Retain the final JSON
and exact rank configuration with each result.

## Required throughput sample

Every baseline and candidate run must retain all three throughput views:

- `effective_prefill_tok_s`: request prompt tokens divided by that request's
  TTFT. It includes scheduler waiting and is therefore the user-visible prefill
  rate, not a pure prefill-kernel measurement.
- `decode_tok_s` and `decode_tok_s_sum`: each request's active decode rate and
  their sum. These show how fast admitted decodes run.
- `aggregate_decode_tok_s`: all post-first-token output tokens divided by the
  interval from the first request's first token until the last request ends.
  This includes mixed-workload stalls and is the appliance delivery rate.

`real_workload_matrix.py` sets `throughput_sample_complete` and exits nonzero if
any required throughput value is missing. A candidate is not eligible for an
optimization commit when this gate fails, even if TTFT improves.
