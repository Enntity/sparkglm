# Selected NVIDIA package, September 9

The NVIDIA + DFlash2 package is available on the package branch with
[its own launcher and exact pins](../docs/NVIDIA_PACKAGE.md). Fresh traditional
C4 video medians: NVIDIA 90.500s, Red Hat 89.243s, three retained runs each.
The 1.41% wall-time difference is smaller than the run-to-run spread. This is
bounded package-selection evidence, not G3/G5 promotion or a quality claim.
[Full comparison and all samples](candidates/2026-09-09-nvidia-redhat-video/RESULT.md).
The existing main-branch default and historical measurements follow below.

# Current public default: NVFP4 512K research preview

As of September 8, the maintainer selected the measured NVFP4 mixed recipe as
the public install default. This does not upgrade any recorded gate or claim
G5 certification. The prior preview status below is retained as history.

| Frozen matched suite (three retained repetitions) | Reference EXL3 | Updated EXL3 | NVFP4 |
| --- | ---: | ---: | ---: |
| C1 16K TTFT, seconds | 11.056 | 10.949 | 8.911 |
| C1 32K TTFT, seconds | 21.519 | 24.884 | 17.113 |
| C4 16K complete wall, seconds | 78.54 | 67.31 | 57.21 |

[Checksum-bound comparison](candidates/2026-09-07-tp2-comparison/RESULT.md).
These compare complete tuned configurations: EXL3 used 1M context, NVFP4 64K.
TTFT includes first-token work; it is not a raw prefill-kernel throughput test.
Bounded semantic results were 15/16 reference, 14/16 updated EXL3, 15/16 native
NVFP4, and 16/16 slower Marlin. They do not establish broad quality equivalence.

The subsequent **512K / 9 GiB** profile passed a 523,264-token cold prompt
(TTFT 341.758s) and two cached tool continuations (~1.9/2.1s).
C4 at 64K each passed all 12 turns, but cached latency under contention ranged
7.4–151.3s with preemption. [Capacity evidence](candidates/2026-09-08-nvfp4-context-512k9/RESULT.md).
The previous 1M attempt failed with insufficient memory; 512K is a deliberate
headroom tradeoff, not a full-window concurrency guarantee.

The **separate staggered video workload** uses four ~16K field-guide prompts,
arrivals 0/1/2/3s, and 400 output tokens per request. Median of three:

| Recorded configuration | Complete wall | Output tokens / complete wall |
| --- | ---: | ---: |
| NVFP4 mixed, 512K | 83.869s | 19.08 tok/s |
| NVFP4 skip, 512K | 116.122s | 13.78 tok/s |
| Mia 9c0794b default skip, 850K | 128.176s | 12.48 tok/s |

All 12 requests per configuration completed. The NVFP4 mixed campaign had
preemptions; neither skip campaign did. Capacities, drafters, chunks, and
scheduler implementations differ. These are complete recipe comparisons,
not an isolated quantization experiment or criticism of Mia's work.

Bundles: [mixed](candidates/2026-09-08-nvfp4-c4-video/RESULT.md),
[skip](candidates/2026-09-08-nvfp4-skip-c4/RESULT.md),
[Mia as presented](candidates/2026-09-08-mia-9c0794b-c4/RESULT.md).

---

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
