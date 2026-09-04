# SparkGLM

SparkGLM is an open research project for running **GLM-5.3-Flash on exactly two
NVIDIA DGX Spark GB10 systems**. The current runnable path is an opinionated
vLLM/EXL3 appliance derived from MiaAI-Lab's two-Spark recipe. The repository
also preserves the earlier Atlas-native implementation and the experiments
that succeeded, failed, or changed the direction of the project.

> **Release-candidate status:** this repository is staged privately for a
> publication review. It has not been declared production-ready, and no
> repository visibility change is part of this commit.

## How changes earn promotion

```text
idea -> static checks -> exact-shape operator -> tinyGLM
     -> full 16K/32K C1/C2 comparison -> quality/reliability -> default
```

This promotion path is the center of the project. tinyGLM is the mandatory
fast integration gate: it preserves the production kernel geometry without a
164 GiB model load. It decides whether a candidate deserves full-model time;
it never substitutes for real-model performance or quality evidence.

Read [the test methodology](docs/METHODOLOGY.md), then browse the canonical
[current qualification status](results/CURRENT.md) and the complete
[results and qualification index](results/README.md). Every performance claim
must point to a checksum-bound `qualification.json`. Pre-policy results are
kept as explicitly legacy evidence rather than retroactively certified.

## Start here

- **Check the component licenses before downloading:** read
  [docs/LICENSING.md](docs/LICENSING.md). The fastest default downloads a
  CC BY-NC-ND 4.0 DFlash2 checkpoint; `SPEC_METHOD=mtp` and `none` avoid it.
- **Run the current engine:** follow the build and two-node launch process in
  [SPARKGLM.md](SPARKGLM.md) and the detailed upstream recipe guide in
  [docs/upstream/MIA_RECIPE_README.md](docs/upstream/MIA_RECIPE_README.md).
- **Understand what is original:** read
  [docs/PROVENANCE.md](docs/PROVENANCE.md) and
  [docs/ATTRIBUTION.md](docs/ATTRIBUTION.md), then consult the practical
  [licensing boundaries](docs/LICENSING.md).
- **Inspect the evidence:** see [docs/RESULTS.md](docs/RESULTS.md), retained raw
  receipts and reports under `results/`, and the code archive under
  `research/vllm-iterations/`.
- **Inspect the native-engine attempt:** see `research/atlas/`. It is valuable
  research, but it is not the recommended serving path.
- **Review before publication:** see
  [docs/PUBLICATION_REVIEW.md](docs/PUBLICATION_REVIEW.md).

## What is running code versus research

| Path | Status | Purpose |
| --- | --- | --- |
| repository root | current candidate | Two-Spark vLLM + EXL3 serving recipe and optimized kernels |
| `benchmarks/` | test harnesses | Reproducible endpoint, tinyGLM, and kernel A/B programs |
| `results/` | canonical evidence | Indexed qualification records, reports, raw receipts, limitations, and rejected work |
| `research/current-engine-history/` | provenance | Accepted commit mailbox without unsafe historical git objects |
| `research/vllm-iterations/` | historical | Accepted and rejected vLLM-era experiments, measurements, and patch mailboxes |
| `research/atlas/` | archival | AGPL Atlas GLM implementation, probes, and a reconstructable source patch |

The project deliberately retains negative results. A rejected patch is not an
optional optimization and should not be enabled merely because its source is
available.

## Current measured picture

All numbers below were measured on the project's two directly connected GB10
systems. They are not vendor claims.

- Enabling work-conserving mixed scheduling instead of strict prefill deferral
  raised four-stream 16K aggregate output rate by **44.2%**, reduced makespan by
  **25.0%**, and cut the fourth request's TTFT by **50.7%**. This is a policy
  correction using existing scheduler machinery, not a 44% kernel speedup.
- Against the prewarmed Mia mixed control at `eb0469f`, the earlier SparkGLM
  candidate `5940f05` raised aggregate output rate by **7.4%** and reduced wall
  time by **5.7%** on the retained four-stream 16K workload.
- The later GPU-resident grouped-prefill path improved full-model effective
  prefill by **9.6% at C1** and **6.3% at staggered C2** on approximately 33K
  actual-token prompts; C2 wall time improved **5.9%**.
- The cooperative decode kernel improved the isolated exact-shape kernel by
  **5.5-13.9%** and tinyGLM endpoint throughput by **5.5% at C1** and **2.9% at
  C4**. Long C2 was flat. It still needs a broader full-checkpoint quality and
  workload gate before a universal production claim.

See [docs/RESULTS.md](docs/RESULTS.md) for scopes and primary receipts.

## What is not included

This repository contains no model checkpoints, EXL3/TR3 weight files,
DFlash2 weights, abliteration direction tensors, API credentials, private SSH
keys, compiled CUDA shared objects, Python caches, or machine-local `.env`
files. Downloaded artifacts remain under their own licenses and are fetched by
the operator from their original sources.

Generated comparison videos are also excluded from git. The raw JSON traces
and the rendering harness are retained so videos can be regenerated without
turning the source repository into a media archive.

## Hardware and scope

The optimized path is intentionally narrow:

- 2x NVIDIA DGX Spark / GB10 / SM121
- GLM-5.3-Flash
- TP=2
- EXL3/TR3 K4 routed experts
- DFlash2 k=7 where its separate license and workload fit are acceptable
- medium and long staggered workloads, not only short synthetic decode

Fallbacks and rollback knobs remain because a fast unsupported shape is a bug,
not an optimization.

Important appliance defaults include work-conserving
`GLM53_MIXED_PREFILL_CHUNK=0`, the GB10-selected 16 ms TP spin window, and the
`rightsize` mode for `GLM53_INDEXER_WORKSPACE`. Cooperative EXL3 decode remains
an explicit opt-in until its full-checkpoint workload and quality gate is
broader; see `.env.example` and [SPARKGLM.md](SPARKGLM.md) for the complete
controls.

## Licensing

SparkGLM is a multi-license repository because it preserves work from several
upstreams:

- Original project integration: Apache-2.0 unless an explicit path rule says
  otherwise.
- MiaAI-Lab serving-kit material and retained modifications: MIT; mixed vLLM
  patchers also retain Apache-2.0 obligations.
- Atlas-derived source and patches under `research/atlas/`: AGPL-3.0-only.
- Upstream FlashKDA source: MIT. SparkGLM's Atlas bridge is AGPL, while the
  slot patch contains MIT-derived context plus AGPL modifications.
- Other third-party material retains its file-level license and attribution.
- The bundled Z.ai chat template is covered by the GLM-5.3 License.

Read [LICENSE](LICENSE), [NOTICE](NOTICE), and
[docs/LICENSING.md](docs/LICENSING.md) before redistribution. The optional
DFlash2 checkpoint is CC BY-NC-ND 4.0; use `SPEC_METHOD=mtp` or `none` when
those terms do not fit the deployment.

## Publication gate

Run:

```bash
./scripts/check.sh all
```

Passing that script is necessary but not sufficient. A human should still
review the attribution table, benchmark wording, excluded-artifact list, and
every file named in `docs/PUBLICATION_REVIEW.md` before changing repository
visibility.
