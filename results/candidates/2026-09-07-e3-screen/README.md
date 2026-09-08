# E3 first kernel screen — universal dispatch rejected

The unmodified E3 arithmetic passed the declared reference-relative numerical
checks for all 17 cases, including changed-input CUDA graph replay and output
redzones. It failed the universal performance screen: small cases regressed,
while the larger multi-expert cases improved by approximately 25–33%.
This is synthetic kernel evidence, not full-model throughput or quality.
The 288-expert fixture has unrealistically overlapping routes and is retained
as a stress case, not presented as a production router distribution.

Follow-up hypothesis, declared after seeing this screen: retain the existing
thin/decode and grouped path below 2048 input tokens; enable E3 only above that
threshold. Add production-like top-8 row counts, empty/nonlocal sentinel cases,
and sanitizer checks before tinyGLM. The opt-in adapter is disabled by default.
A new result bundle must qualify that policy; this failed universal screen does
not qualify it. G2–G5 remain outstanding.
