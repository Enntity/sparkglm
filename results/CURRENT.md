# Current qualification status

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

The legacy labels matter. The old “16K” isolation generator produced roughly
33K actual tokenizer tokens in at least the grouped-prefill campaign, the Mia
comparison retained only two repetitions per arm, and none of these bundles is
a complete current G3 plus G4 matrix. See `docs/KNOWN_LIMITATIONS.md`.
