# Reederey reference findings

This document records what the pinned Reederey production repository changes
about SparkGLM's starting point. These are that project's measurements on its
two-Spark pair, not measurements from this repository. Every performance choice
still has to survive SparkGLM's 16K/32K staggered C1-C2 gate.

## Changes worth carrying into the source baseline

- Mixed-prefill gate v3 replaces an indefinite `skip` with bounded admission,
  a warm-tail bypass, and aging from 512 toward 1,792 tokens. Their matched
  window reduced a 60K peer read from 78.3 to 63.1 seconds and increased decode
  work in a 240-second window from 677 to 863 tokens while short TTFT remained
  6.65 seconds. We carry the mechanism; we will retune its constants.
- Per-group APC retention stops the short DFlash sliding-window group from
  consuming most global cache IDs for state that does not determine the hybrid
  cache hit. The change composes with Mia's hybrid-hit correction and is runtime
  configurable, so it belongs in the baseline.
- The E2 EXL3 fat-expert path improved their cold reads by 9-18 percent across
  178K-254K prompts with decode essentially flat. That is independent evidence
  for retaining Mia's kernel while we reproduce it at 16K/32K.
- The alignment-floor guard fixes a zero-progress edge when a mixed-prefill cap
  falls below a cache page. It is a correctness guard, even though the default
  1,792-token threshold keeps the edge dormant.

## Settings that remain A/B questions

- Mia defaults DFlash draft TP to 2; Reederey measured TP2 as a 1-2 percent loss
  and retained TP1. We will measure both with the same image and prompt stream.
- Mia moved the batch budget to 7,168; Reederey found it safe but neutral or
  slightly worse under its scheduler and retained 3,584. The budget and the
  long-prefill threshold are separate controls and must not be conflated.
- Reederey pins DFlash2 source revision `7d74cdd881ed7e32c31175984a67823127b66cfe`;
  later revisions did not win in its tests. We will compare the pin against the
  current recipe revision rather than silently floating the dependency.
- Fine-grained APC materially improves sub-page follow-ups but carried about a
  2.7 percent structured-decode tax in their measurement. It needs a separate
  warm/fork gate and remains optional in the first source baseline.
- A 2 ms spin wait reduced CPU and temperature without a throughput loss there.
  It is an operational candidate, not a claimed GPU speedup.

## Rejected assumptions we will not repeat

- A static 3,584-token long-prefill threshold improved solo prefill by 9.7
  percent but made a short request wait roughly 280-298 seconds instead of about
  seven seconds. Solo prefill is not the product objective.
- A larger prefill chunk can move the memory low-water mark from the head Spark
  to the worker. Both ranks must be sampled; head-only free memory is not a safe
  gate.
- The router-GEMM overlay in the reference is still a candidate, not an adopted
  measured win. It stays out of our baseline. Atlas also found that a small
  router speedup could reduce speculative acceptance and lose end to end.
- Cache-reset and benchmark endpoints are operational tooling. They must not be
  added to the serving surface without authentication and an explicit need.
