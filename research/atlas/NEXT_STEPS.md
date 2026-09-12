# Atlas continuation: bounded next work

The [day-two bundle](../../results/candidates/2026-09-11-atlas-day2/RESULT.md)
contains the current evidence and fixed failed gates. Keep Atlas experimental
and separate from the vLLM default. No new route is already qualified merely
because this source is committed or a PR is merged.

## Rebuild the measured configuration honestly

Reconstruct the source, embedded kernels and tests from the published lineage.
Compile the separately pinned FlashKDA dependency and reviewed bridge. Preserve
CUDA 13/sm_121a and arithmetic flags, tests with positive counts, checked arena
spans, pre-KV library loading, four-owner 32K capacity and memory guards.

The optional native sparse continued-prefill bridge depends on NVIDIA object
SHA256 `9e372b5a47ade0a8332a73878ec3f6b917ae137447fea2335d2003dd146d549a`.
Its exact source identity is unproven. Establish a licensed immutable source
recipe and independently qualify its build before claiming complete reproducibility
of that measured path. The compiled object is intentionally not distributed.
Use the source-based BF16 fallback when that optional artifact is unavailable;
measure it as its own configuration.

## Choose the next experiment from bounded evidence

- **Within-owner MLA query batching:** Qa, Qb and index-Q remain row-serial in
  repaired K3, accounting for about 7.141 ms per sampled target. First compare
  existing M3 batchm against three scalar operations on distinct inputs. Keep
  each row's normalization and causal cache/index/attention updates unchanged.
  Integration needs a checked scratch partition and protection against the
  current index-Q overwrite. This is a source-feasibility candidate, not a win.
- **Joint-owner routed work:** replayed actual routes passed an inclusive
  routed-stage screen, but the engine still verifies owners serially. A cohort
  implementation needs stable sequence generations, per-owner canonical KV and
  SSM state, causal 32K indexing, rejection repair, both-rank command symmetry,
  and correct final/partial-owner handling. Do not loosen old paired-mode bounds
  or call queued C1 execution C4. Stage stateless projection/FFN factoring before
  touching target state. Compare against four serial owners and test every
  accepted-prefix/rollback case. Replay gains apply to routed work, not the whole
  MoE, target or C4 interval.
- **Speculation:** native MTP4 reuses more existing code than porting the vLLM
  DFlash2 draft, but long-context C4 state/metadata and capacity support still
  need implementation and qualification. MTP3 is not an existing allowlisted
  configuration. DFlash2 requires missing draft operations/loader semantics,
  precision/layout compatibility, long sparse verification and ownership work.
  The old vLLM MTP1 high-acceptance screen used different prompts. Obtain matched
  first-position evidence before blaming an Atlas head defect.

Do not reopen the rejected router, compact-down, matrix-TP, tile or fusion
variants merely by lowering their recorded gates. Require a materially new
mechanism and a cheap exact-shape screen first. Existing C1 profiles cover one
rank; NCCL time can include peer waiting and is not all recoverable transfer.

## Qualify the complete appliance

Before claiming a release or default win, run the documented tinyGLM integration
and full alternating workload matrix with actual tokenizer counts, then relevant
quality/operational tests. Preserve original failures, warmup rules, stream event
gaps, TTFT and total wall separately. Use per-position speculative counters or
the explicitly limited audit; zero rejected-token usage is not acceptance proof.
Full-model changed continuations and unchanged-image variability require care
when attributing a small wall-time difference.

Keep benchmark/harness source portable and close its dependency imports. Private
cache images, source paths or laboratory endpoints must not become required
public defaults. Source and selected results belong in the single active Atlas
line; raw traces, tensors, provider reasoning, binaries and deployment config
remain in the private evidence archive.
