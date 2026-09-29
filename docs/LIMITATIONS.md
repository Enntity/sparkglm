# Limitations

What to know before relying on SparkGLM.

## Scope of the measurements

- We tested one pair of DGX Sparks on one cable. Other firmware, drivers,
  cables or cooling may give different numbers.
- The vLLM and Mia comparison runs were recorded earlier on the same pair.
  They were not alternated with the Atlas runs.
- The published numbers used the recipe as it stood before the repository was
  reorganized. That recipe used the same engine commit and profiles.
- A cold image build has not been timed. Our builds reused BuildKit caches.

## Not yet measured with the recipe image

- **Long context:** we have not measured it with the recipe image. On the same
  engine commit, before the thinking-budget fix, four concurrent 190K requests
  and one 500K request completed with at least 8.3 GB of host memory free.
- **Endurance, cancellation under load, and multimodal work at full context:**
  not measured.

## Capacity

The `4x512k` profile admits four requests of up to 512K each. They share one
KV pool of about 866K tokens, so four full-length requests at once do not fit.
Later requests wait for room.

## Performance gaps

We compared against the published RiNGSiDE vLLM TP2 results, not a run on our
pair. Atlas is behind on:

- single-stream decode: about 7% on code and 20% on prose;
- low-concurrency short code: 36.9 against 44.0 tok/s with one stream, 76.7
  against 97.3 with six;
- long cold prefill: 28.3 against 25.7 s at 64K.

Atlas is ahead on newcomer time-to-first-token under load and on 8K prefill.

## Behaviour

- Requests with thinking disabled still get a short, low-effort reasoning
  block, returned in `reasoning_content`. The reasoning budget is enforced, so
  structured answers are not trapped inside reasoning.
- The quality probe scored 11/12 on two-hop needles in the recorded run; the
  same engine scored 10–12/12 across earlier runs. Arithmetic scored 40/40.
- Sparse attention uses GLM-5.3's full 2051-candidate selection. The vLLM
  version's 2048-candidate approximation does not apply here.

## Operations

- The API has no authentication and listens only on rank 0's loopback.
- GB10 memory is shared with the host. Running other large workloads on
  either Spark while serving can hang the host.
- Both shipped profiles need the DFlash2 drafter, which is licensed for
  non-commercial use only. No profile without it is shipped yet.
- The image is built for arm64 and SM121 (GB10) only.
