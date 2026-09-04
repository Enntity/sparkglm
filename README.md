# SparkGLM

SparkGLM is an open research project for running **GLM-5.3-Flash on exactly two
NVIDIA DGX Spark GB10 systems**. The current runnable path is an opinionated
vLLM/EXL3 appliance derived from MiaAI-Lab's two-Spark recipe. The repository
also preserves the earlier Atlas-native implementation and the experiments
that succeeded, failed, or changed the direction of the project.

> **Release-candidate status:** this repository is staged privately for a
> publication review. It has not been declared production-ready, and no
> repository visibility change is part of this commit.

## Start here

- **Run the current engine:** follow the build and two-node launch process in
  [SPARKGLM.md](SPARKGLM.md) and the detailed upstream recipe guide in
  [docs/upstream/MIA_RECIPE_README.md](docs/upstream/MIA_RECIPE_README.md).
- **Understand what is original:** read
  [docs/PROVENANCE.md](docs/PROVENANCE.md) and
  [docs/ATTRIBUTION.md](docs/ATTRIBUTION.md).
- **Inspect the evidence:** see [docs/RESULTS.md](docs/RESULTS.md), retained raw
  receipts under `benchmarks/receipts/`, and the older campaign under
  `research/vllm-iterations/`.
- **Inspect the native-engine attempt:** see `research/atlas/`. It is valuable
  research, but it is not the recommended serving path.
- **Review before publication:** see
  [docs/PUBLICATION_REVIEW.md](docs/PUBLICATION_REVIEW.md).

## What is running code versus research

| Path | Status | Purpose |
| --- | --- | --- |
| repository root | current candidate | Two-Spark vLLM + EXL3 serving recipe and optimized kernels |
| `benchmarks/` | current evidence | Reproducible endpoint, tinyGLM, and kernel A/B harnesses |
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

- Current project integration and Apache-marked sources: Apache-2.0.
- MiaAI-Lab serving-kit material: MIT.
- Atlas-derived source and patches under `research/atlas/`: AGPL-3.0-only.
- FlashKDA material under `research/atlas/flash_kda/`: MIT.
- Other third-party material retains its file-level license and attribution.

Read [LICENSE](LICENSE), [NOTICE](NOTICE), and
[docs/ATTRIBUTION.md](docs/ATTRIBUTION.md) before redistribution.

## Publication gate

Run:

```bash
./scripts/publication-audit.sh
python3 tests/test_indexer_workspace.py
python3 tests/test_tinyglm.py
python3 tests/test_tinyglm_gate.py
```

Passing that script is necessary but not sufficient. A human should still
review the attribution table, benchmark wording, excluded-artifact list, and
every file named in `docs/PUBLICATION_REVIEW.md` before changing repository
visibility.
