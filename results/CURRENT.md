# Current qualification status

**Source-restoration correction:** the first release reconstruction omitted
substantive code inherited from the posted video's custom vLLM foundation.
The default build now restores that implementation, not just
its flags. See [the source audit](../docs/VIDEO_SOURCE_RESTORATION.md). Earlier
reconstruction timings are not evidence of implementation parity.

The new [source-restoration evidence](candidates/2026-09-04-video-source-restoration/RESULT.md)
records exact source checks on both ranks, native oracle checks, tinyGLM
results including the initial failed throughput guard, and the new full-model
C4 replays. Implementation identity is not a claim of binary reproducibility,
deterministic full-model text, or a new speedup. The bundle is not promoted.
Its first restored-image C4 median was 90.678 seconds versus 87.552 seconds for
a contemporary retained-original-image run, a 3.57% wall-time increase. The
video-style aggregate rates were 22.770 versus 24.023 tok/s. Those historical
receipts remain intact; the initial small-sample gap was not causally isolated.

**Stronger controlled follow-up: headline speed is effectively tied.**
[Ten measured repetitions per image](candidates/2026-09-04-video-runtime-isolation/TEN_REP_RESULT.md),
across two warmed startups each in reversed order, give mean wall times of
88.835 seconds preserved versus 88.907 seconds rebuilt (**+0.08%**), and mean
delivered rates of 23.372 versus 23.458 tok/s. The earlier 5% throughput deficit
did not reproduce with matched cache/CPU controls. Secondary latency differences
and conditional uncertainty are reported, not omitted. This remains one C4
workload, not complete G3/G4/G5 or proof of equality under automatic sizing.

The follow-up [runtime and native-artifact investigation](candidates/2026-09-04-video-runtime-isolation/RESULT.md)
checks the whole package environment, input token IDs, CUDA instruction bodies,
and bounded operator timings. It distinguishes confirmed environment drift
from a causal explanation; neither a missing library nor a speed fix is assumed.

The runnable engine predates qualification-v1. Its strongest retained evidence
has been migrated honestly rather than re-labeled as newly certified:

- [`2026-09-03-current-best-posted-video`](legacy/2026-09-03-current-best-posted-video/)
  is the exact timing receipt and identity record for the later
  `4b237597+c805318` configuration now selected by fresh-checkout defaults;
- [`2026-09-03-grouped-prefill-k4`](legacy/2026-09-03-grouped-prefill-k4/)
  contains the three-pair full-checkpoint prefill result and raw receipts;
- [`2026-09-03-latest-mia-apples-to-apples`](legacy/2026-09-03-latest-mia-apples-to-apples/)
  contains the warmed four-stream Mia comparison;
- [`2026-09-03-cooperative-decode`](legacy/2026-09-03-cooperative-decode/)
  contains the exact-shape and tinyGLM cooperative-decode evidence.

No result currently claims post-policy G5 release qualification. Selecting the
posted configuration as the reconstructable default does not grandfather its
legacy evidence; the next performance claim must establish a clean, signed
baseline under the new methodology.

The first post-policy reconstruction measurement is now available at
[`2026-09-04-clean-release-c4-replay`](candidates/2026-09-04-clean-release-c4-replay/).
The clean `b3d0bbd` image completed three warmed C4 16K-class repetitions at a
23.514 aggregate decode tok/s median, 3.1% below the retained posted one-off.
It is deliberately labeled G0 baseline evidence: only this workload was run,
there was no paired arm, and no G4 quality suite was performed. Its source
receipt records file differences; the later audit identified substantive
omissions, now explicitly acknowledged in that bundle. Its numbers apply to
the incomplete reconstruction only.

The legacy labels matter. The old “16K” isolation generator produced roughly
33K actual tokenizer tokens in at least the grouped-prefill campaign, the Mia
comparison retained only two repetitions per arm, and none of these bundles is
a complete current G3 plus G4 matrix. See `docs/KNOWN_LIMITATIONS.md`.
