# Current qualification status

## September 7 TP2 experiments

- [Managed full-model E3 comparison](candidates/2026-09-07-e3-managed/qualification.json)
  retains the broad E3 policy as rejected: concurrent gains did not protect
  isolated latency.
- [Concurrent E3 and 32-row screens](candidates/2026-09-07-e3-concurrent-screen/qualification.json)
  preserve passing operator checks and the tiny-model decode regression.
- [Cooperative decode through 32 tokens](candidates/2026-09-07-decode32-screen/RESULT.md)
  found small mixed gains; the serving limit remains 16.
- [MXFP8 draft support](../research/experiments/mxfp8-draft/README.md)
  is self-contained and opt-in. The inherited loader uses actual draft TP2;
  a TP1 flag does not create an independent draft group.

These experiments do not change the reconstructable default below. Final
appliance comparisons may independently tune each path, while preserving
quality and reporting the complete configuration and context capacity.

The root build targets the final posted-video engine and configuration,
including grouped prefill and cooperative decode. See the
[runtime mapping](../docs/PUBLISHED_VIDEO_CONFIGURATION.md).

Release verification found effectively matching headline performance between
the preserved image and the rebuilt engine in a controlled, warmed C4 replay
(ten repetitions per image). This checks that we are releasing the intended
implementation; it is **not a new optimization claim or full G3/G4/G5
qualification**. Detailed checks, earlier partial results, and limitations
remain in the [verification record](candidates/2026-09-04-video-runtime-isolation/TEN_REP_RESULT.md).

## Retained optimization evidence

The engine predates qualification-v1. These records are explicitly `legacy`,
not retroactively certified:

- [`2026-09-03-current-best-posted-video`](legacy/2026-09-03-current-best-posted-video/):
  timing receipt and identity for the `4b237597+c805318` video configuration.
- [`2026-09-03-grouped-prefill-k4`](legacy/2026-09-03-grouped-prefill-k4/):
  three-pair full-checkpoint prefill measurements and raw receipts.
- [`2026-09-03-latest-mia-apples-to-apples`](legacy/2026-09-03-latest-mia-apples-to-apples/):
  warmed four-stream Mia comparison.
- [`2026-09-03-cooperative-decode`](legacy/2026-09-03-cooperative-decode/):
  exact-shape and tinyGLM evidence, not a standalone full-model speedup.

The old “16K” isolation generator produced roughly 33K actual tokens in at
least the grouped-prefill campaign; the Mia comparison retained only two
repetitions per arm. No bundle certifies the complete current G3 plus G4
matrix or G5 release. See [known limitations](../docs/KNOWN_LIMITATIONS.md).

Selecting the posted configuration as the reconstructable default does not
grandfather its evidence. New performance claims must use the
[methodology](../docs/METHODOLOGY.md) and checksum-bound qualification records.
The [complete index](README.md) retains negative and partial experiments too.
