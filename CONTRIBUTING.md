# Contributing to SparkGLM

We optimize GLM-5.3-Flash on **two DGX Spark GB10 systems**. Small, scoped
contributions are welcome: documentation, tests, launch reliability, kernels,
and measured improvements. You do **not** need Sparks to submit a PR.

Want to run the real model instead? Use [the recipe](SPARKGLM.md).

## First contribution: no GPU required

Use Git and Python 3.11 (the source-CI version). Fork/clone the repo, then:

```bash
git switch -c my-change
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
./scripts/check.sh all
```

This runs only source-safe tests, licensing checks, and evidence validation.
It does not SSH to a Spark, build CUDA, or start a model. The same dependency
file and command are used by CI.

Make one focused change, add a regression test where appropriate, rerun the
command, and open a PR using the template. Documentation-only changes normally
need only this G0 gate. Mark hardware gates **not run / maintainer needed**,
not passed. A maintainer reviews the exact commit before running private
hardware tests. Do not add automatic private-Spark access to public CI.

For runtime work, read [the gate definitions](docs/METHODOLOGY.md) and follow
the experiment loop below. You can submit an unqualified experiment for
review without hardware; merging runtime changes still requires the applicable
gates. Nobody should fabricate a hardware result to get a PR reviewed.

## Where to make a change

| Area | Start here |
| --- | --- |
| Serving/configuration | `start.sh`, `.env.example` |
| EXL3 kernels and dispatch | `overlay/exl3*.cu`, `overlay/exl3*.cuh`, `overlay/exl3.py` |
| vLLM runtime/native changes | `patches/video/`, `patches/runtime/`, root `Dockerfile` |
| Fast integration fixture | [tinyGLM](docs/tinyglm.md), `scripts/tinyglm.sh` |
| Benchmark collection | `benchmarks/`, `scripts/full-model-gate.sh` |
| Evidence format | [qualification records](docs/QUALIFICATION.md), `results/` |
| Older work, not the serving path | `research/`; Atlas is AGPL and archival |

The [provenance ledger](provenance/upstreams.json) and
[licensing rules](docs/LICENSING.md) govern source reuse. Do not copy Atlas
code into the Apache serving path without resolving that license boundary.

## Runtime experiment: baseline → candidate → evidence

Use a reserved Spark pair and the network configuration from
[the quickstart](SPARKGLM.md). This sequence replaces the normal recipe's
containers; stop production through its own manager first.

### 1. Record the reference with tinyGLM

Start from a clean reference checkout and preserve the reference image tag:

```bash
scripts/tinyglm.sh restart
scripts/tinyglm-gate.sh record .tinyglm-gates/baseline.json
```

The fixture metadata builds in under a second; an uncached **engine image**
still needs native compilation. tinyGLM avoids the full model load, not the
compiler. Its [guide](docs/tinyglm.md) states the reduced architecture, cases,
and limitations. Do not mistake its generated text or tok/s for real GLM quality
or speed.

### 2. Build your candidate explicitly

The normal Dockerfile verifies the frozen video source manifest. **Do not edit
that manifest to make an experiment pass.** Follow
[the candidate-build guide](docs/CANDIDATE_BUILDS.md) to declare exact changed
file hashes and reasons in a separate candidate manifest.

For an unchanged-source dry run, the shipped empty declaration is runnable:

```bash
python3 scripts/build_candidate.py \
  --manifest provenance/candidate-sources.example.json \
  --tag sparkglm-candidate:my-change --print-dockerfile
```

For actual changes, use your own declaration as described in that guide.
Commit your source, tests, and declaration first. Then build with the same
command without `--print-dockerfile`. Only `sparkglm-candidate:NAME` tags
are accepted; this does not replace `sparkglm:local`.

Build while resident models are stopped and at least 32 GiB is available.
The helper only builds; it does not stop or launch anything for you.

### 3. Run and compare

```bash
IMAGE=sparkglm-candidate:my-change SKIP_BUILD=1 SKIP_PULL=1 \
  scripts/tinyglm.sh restart
scripts/tinyglm-gate.sh compare \
  .tinyglm-gates/baseline.json .tinyglm-gates/candidate.json
```

Here `SKIP_BUILD=1` deliberately selects your already built candidate; it is
not an instruction for consumer/reference installs. Both rank images are
shipped/checked by the launcher. Check the reported image before recording.

The paired gate checks token IDs, completion, determinism, and regressions
across C1, staggered C4, and long C2. Operator changes also need G1 exact-shape
oracle/graph tests. G2 passes only earn full-model testing time.

### 4. Record what was actually proved

From a clean committed candidate, create a result skeleton:

```bash
python3 scripts/qualification.py new \
  --id my-first-experiment \
  --title "My first experiment" \
  --target "32K staggered C2 TTFT" \
  --baseline-ref BASELINE_COMMIT
```

Replace `BASELINE_COMMIT` with your recorded reference SHA. Put raw evidence
inside that bundle, then fill in the actual gates, configuration, results,
limitations, and attestation. Do not mark a candidate promoted.

For endpoint claims, use the [full-model collection walkthrough](docs/QUALIFICATION.md):
it specifies both image digests, model revision, pair ID, exact-token
16K/32K C1/C2/C4 matrix, and alternating baseline/candidate repetitions.
The collector does not switch images or statistically qualify the result for
you. Quality and operational checks are also required when relevant.

```bash
python3 scripts/qualification.py checksum \
  results/candidates/my-first-experiment/qualification.json
python3 scripts/qualification.py verify-all
./scripts/check.sh all
```

Negative and partial results are welcome when clearly labeled. Preserve the
raw baseline/candidate outputs and source declaration. Tiny or noisy gains
must not be promoted from a best sample.

## Commit and PR checklist

- One logical improvement per commit; name its target, protected behavior,
  rollback, and gate level.
- Preserve upstream copyright/notice files. Identify copied, adapted, ported,
  or inspired work and its exact source revision; update the provenance ledger.
- Follow [commit provenance](docs/PROVENANCE.md): `Provenance:`,
  `Original work:`, and `Verification:`. Use
  `Provenance: original implementation` when no external implementation was
  incorporated. Credit is not the same as co-authorship.
- Do not commit weights, tensors, compiled binaries, private paths, credentials,
  `.env`, or generated videos.
- Run G0 for every PR; report relevant G1–G5 results or honestly state what
  remains untested. For runtime work, also run container tests inside the built
  image with the checkout mounted; see [candidate builds](docs/CANDIDATE_BUILDS.md).
- Update [known limitations](docs/KNOWN_LIMITATIONS.md) when semantics or
  supported scope changes. Keep unqualified changes opt-in.
- Do not claim speed or quality certification from checksums alone. Promotion
  requires maintainer review and the gates in [the methodology](docs/METHODOLOGY.md).
