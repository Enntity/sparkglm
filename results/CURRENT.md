# Current qualification status

## September 7 TP2 experiments

The native NVFP4 branch now has real two-Spark execution evidence, beyond the
earlier copied recipe preparation. These remain explicit experiments:

- [Native FP4 operator and tiny-model checks](candidates/2026-09-07-nvfp4-managed/qualification.json)
  establish the selected compute path and fixture behavior.
- [Tuned NVFP4 with MXFP8 draft](candidates/2026-09-07-nvfp4-mxfp8-1k/RESULT.md)
  records three retained full-matrix repetitions at actual target/draft TP2,
  64K context, 1K prefill chunks, and 9 GiB KV per rank.
- [Native NVFP4 with 2K chunks](candidates/2026-09-07-nvfp4-native-2k/RESULT.md)
  is the final overall NVFP4 tuning choice: three retained matrices improve
  four case medians over 1K and approximately tie the primary C2-32K case.
  It retains the same single bounded arithmetic failure.
- [Initial full-model screen](candidates/2026-09-07-nvfp4-full-screen/RESULT.md)
  preserves the BF16-draft comparison and cache-admission failure.
- [Humming fixture screen](candidates/2026-09-07-humming-fixture/RESULT.md)
  remains unqualified for full serving. Its installed path falls back to FP8
  activations and is distinct from native FP4 compute.
- [Marlin operator and TP2 fixture checks](candidates/2026-09-07-marlin-managed/RESULT.md)
  qualify the W4A16 alternative for full-model time, including independent
  numeric, clamp, redzone, and sanitizer evidence.
- [Marlin full-model screen](candidates/2026-09-07-marlin-full/RESULT.md)
  passes all 16 bounded semantic checks but does not beat native NVFP4 on the
  primary 32K two-request workload. It remains a separately recorded W4A16 option.
- [Independent configuration comparison](candidates/2026-09-07-tp2-comparison/RESULT.md)
  retains the reference and screened EXL3/MXFP8 alternatives alongside NVFP4,
  with repetition counts, immutable identities, and capacity differences.

The tuned run passed completion/isolation and managed gateway checks but
failed one bounded arithmetic case, also failed by the EXL3 reference. No
general quality, endurance, or default-promotion claim is made. Independent
tuning compares complete configurations, including their context capacity.

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
