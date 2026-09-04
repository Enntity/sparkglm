# Test and promotion methodology

SparkGLM optimizes one appliance: GLM-5.3-Flash on two directly connected DGX
Spark GB10 systems. A change is not "better" because one kernel or one prompt
got faster. It is better only when its declared target improves and the
protected behavior of the appliance does not regress.

## The promotion path

```text
idea
  -> G0 static checks
  -> G1 exact-shape operator checks
  -> G2 tinyGLM integration checks
  -> G3 full-model workload comparison
  -> G4 semantic and operational checks
  -> G5 release/endurance qualification
  -> promoted default
```

An experiment may be retained at any level, including a failed experiment.
Only a qualified change may alter the recommended defaults or support a public
performance claim.

## Change states

- **experiment**: exploratory work, normally disabled and kept under
  `research/` or behind an explicit flag;
- **candidate**: passed the relevant fast gates and is eligible for full-model
  time, but is not a recommended default;
- **accepted**: passed its declared qualification level and is retained, but
  may still be opt-in when its scope is narrow;
- **promoted**: passed every gate required for its change class and may become
  a default;
- **rejected**: measured, documented, and deliberately not enabled.

## Gate definitions

### G0: static and publication-safe

Run `scripts/check.sh quick`. It checks source-only regressions, Python syntax,
shell syntax, JSON qualification records, attribution pins, and prohibited
artifacts. G0 runs on ordinary public GitHub runners and never touches the
private Sparks. `scripts/check.sh all` means every public-safe check: quick plus
the publication audit. Dependency-heavy image tests use `scripts/check.sh
container`; a running endpoint probe uses `scripts/check.sh live`. The gate
inventory fails if a `tests/test_*.py` file is not assigned to one of those
three lanes.

### G1: exact-shape operator

Required for CUDA, Triton, TileLang, collectives, quantization, or custom
operator changes. Compare the candidate against the current implementation at
the production dimensions. Require:

- exact equality when the operation contract is exact, otherwise a declared
  numeric tolerance;
- representative C1/C2/C4 shapes, boundary shapes, and fallback shapes;
- CUDA-graph capture/replay and deterministic repeated execution;
- no NaN, out-of-bounds, sanitizer, rank, or fallback-policy failure;
- warmups excluded and enough timed iterations to characterize run noise.

A microbenchmark proves an operation. It does not prove endpoint performance.

### G2: tinyGLM

Required for engine, scheduler, graph, TP2, cache, and kernel-integration
changes. tinyGLM preserves the kernel-selecting geometry while avoiding the
full 164 GiB load. Run the paired gate described in `docs/tinyglm.md`.

The standard cases are C1 decode, staggered C4 mixed work, and staggered C2
16K prefill. All streams must complete and remain deterministic. A
performance-only change must preserve token IDs. A deliberate semantic change
must instead add a focused golden regression and explain why token identity is
not the right contract.

tinyGLM answers "is the integration safe enough to spend full-model time?" It
cannot certify real-model speed or quality.

### G3: full-model appliance workload

Required for every performance claim and every default-affecting runtime
change. The primary frozen matrix is:

| Case | Arrival | Prompt | Output | Purpose |
| --- | --- | ---: | ---: | --- |
| C1 | isolated | 16K actual tokens | 400 | medium prefill plus decode |
| C1 | isolated | 32K actual tokens | 400 | large prefill plus decode |
| C2 | staggered | 16K actual tokens each | 400 each | real concurrent service |
| C2 | staggered | 32K actual tokens each | 400 each | primary stress workload |
| C4 | staggered | 16K actual tokens each | 400 each | capacity/fairness guard |

The corresponding arms must use identical model, quantization, drafter,
precision, KV budget, graph sizes, scheduler policy, prompt bytes, arrival
offsets, generation parameters, and warmup. Record actual tokenizer counts;
fixture arguments such as `--prompt-tokens 16384` are not authoritative.

Complete graph capture and the shape warmup, then discard one exact workload
run. Alternate the retained order between baseline and candidate. Use at least
three paired repetitions for a large effect. Claims below 5% require enough
pairs for the confidence interval to exclude zero; otherwise report the result
as neutral/noisy. Never select only the best sample.

Declare one primary metric before running. Protect at least:

- aggregate delivered output tokens/s;
- wall time and per-request TTFT;
- inter-token gap p95/max and request fairness;
- completion, request isolation, errors, and output hashes;
- KV capacity, persistent memory, and startup/graph success.

Small improvements are welcome. They become a win only after the combined
stack passes this same matrix.

### G4: semantics and operation

Required whenever a change can affect output, precision, scheduling semantics,
templates, speculative decoding, or fault behavior. Select the relevant
focused tests and record them in the qualification:

- reasoning on/off and reasoning-effort rendering;
- tool calls, parallel tool results, and structured output;
- multimodal placeholder behavior;
- prompt isolation, prefix hit/miss, cancellation, and EOS/stop handling;
- precision/quantization quality comparisons;
- rollback and partial-rank failure behavior.

Throughput traces with coherent-looking snippets are not a quality evaluation.

### G5: release and endurance

Required for a named public release or a new recommended appliance baseline:

- two clean cold starts and warm restarts;
- exact image digest and source identity on both ranks;
- full G3/G4 matrix from the release commit;
- rollback exercise and partial-node failure check;
- eight-hour mixed-arrival endurance with no leak, deadlock, or quality drift.

## Promotion rules by change class

| Change | Minimum before merge | Minimum before default/claim |
| --- | --- | --- |
| docs, attribution, template test | G0 | G0, plus focused G4 for behavior |
| launcher/configuration | G0 + relevant G2 | G3/G4 if default changes |
| CUDA/operator | G0 + G1 + G2 | G3, plus G4 when numeric behavior changes |
| scheduler/cache/TP | G0 + G2 | G3 + operational G4 |
| quantization/speculation | G0 + G1 + G2 | G3 + quality G4 |
| public release | all relevant prior gates | G5 |

## Public-runner boundary

Untrusted pull-request code must never run automatically on the private DGX
Sparks. Public CI supplies G0. A maintainer reviews the candidate SHA and then
runs G1-G5 locally or through a manually dispatched, protected mechanism. The
resulting qualification bundle is checked in on the contributor's change or a
follow-up qualification commit.

## Result integrity

Every performance claim must point to a `qualification.json` validated by
`scripts/qualification.py`. The record binds the claim to its commits,
configuration, evidence checksums, gate outcomes, limitations, and maintainer
attestation. See `docs/QUALIFICATION.md` and the prominent `results/` tree.
