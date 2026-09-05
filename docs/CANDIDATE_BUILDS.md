# Build an experiment without changing the reference

The normal Dockerfile remains the frozen-reference build. Do not edit its
source manifest to make an experiment pass. The candidate helper generates a
separate Dockerfile and labels its image as an **unqualified candidate**;
the reference Dockerfile, manifest, and default image tag remain unchanged.

## Declare intentional differences

Copy `provenance/candidate-sources.example.json` to a tracked file such as
`provenance/my-experiment.json`. The empty example permits no differences.
For each modified file covered by the frozen manifest, add an entry:

```json
{
  "schema": "sparkglm.candidate-sources/v1",
  "changes": {
    "python": {
      "vllm/path/from/the/reference/manifest.py": {
        "reference_sha256": "REPLACE_WITH_REFERENCE_SHA256",
        "candidate_sha256": "REPLACE_WITH_CANDIDATE_SHA256",
        "reason": "Describe the intentional change and its experiment."
      }
    }
  }
}
```

This illustrates the format; it is not a runnable declaration. Use actual
paths and 64-character hashes from `provenance/video-source-parity.json`:

| Candidate group | Frozen inventory | Bytes to hash |
| --- | --- | --- |
| `python` | `python_files` | Final installed Python after all build-time patchers |
| `native` | `native_source` | Patched native compile inputs at the source gate |
| `exl3` | `exl3_compile_inputs` | EXL3 compile/dispatch inputs with license headers |

The reference hash must match the frozen source or its explicit existing
nonfunctional exception. Candidate hashes refer to the modified file, not
the patch file. Use `sha256sum path/to/file` for directly staged files; for
layered patches, materialize the pinned source and apply the build's
transformations first. Failure reports include observed hashes for diagnosis;
do not approve them without reviewing the actual source diff.

Every change needs a reason. Unlisted differences, missing files, undeclared
extra vLLM Python files, wrong candidate hashes, and stale reference hashes
still fail. New/deleted inventory entries are intentionally unsupported by
this first helper; extending inventory requires tooling review. The running
video-identity check remains reference-only.

## Inspect and build

For an unchanged-source dry run, without Docker or a GPU:

```bash
python3 scripts/build_candidate.py \
  --manifest provenance/candidate-sources.example.json \
  --tag sparkglm-candidate:my-change --print-dockerfile
```

Use your own declaration for actual changes. Commit the source, tests, and
declaration so the image has a clean source identity. On the head Spark, stop
resident models through their own launcher/manager and provide at least
32 GiB `MemAvailable`. Run the same command without `--print-dockerfile`.
Build output is visible in the terminal. The helper does not stop or launch
containers for you.

Only local `sparkglm-candidate:NAME` tags are accepted. Image labels record
the Git revision, candidate profile, and declaration hash. Native/Python
self-tests still run. Matching declared hashes proves the experiment contains
those bytes—not correctness, speed, or video equivalence.

## Test without the full checkpoint

First record the reference tinyGLM baseline as shown in
[CONTRIBUTING.md](../CONTRIBUTING.md). Then:

```bash
IMAGE=sparkglm-candidate:my-change SKIP_BUILD=1 SKIP_PULL=1 \
  scripts/tinyglm.sh restart
scripts/tinyglm-gate.sh compare \
  .tinyglm-gates/baseline.json .tinyglm-gates/candidate.json
```

These explicit skip flags select the already-built local candidate; they are
not consumer installation instructions. The launcher still ships the image
to the worker. tinyGLM preserves the caller's `IMAGE` override even when
`.env` names the reference image.

For dependency-heavy tests, from the head's checkout:

```bash
docker run --rm --gpus all \
  -v "$PWD:/workspace" -w /workspace \
  --entrypoint bash sparkglm-candidate:my-change \
  scripts/check.sh container
```

Run only reviewed code on reserved hardware. These tests may exercise CUDA;
they are not public CPU-only CI. Keep outputs in the qualification bundle.
Changed kernels also need G1 operator checks; endpoint/quality claims require
full-model G3/G4 beyond tinyGLM.

To restore the reference, stop the candidate and use a clean reference
checkout with its preserved configuration and `sparkglm:local` image. Avoid
launching the default tag from modified candidate sources: the launcher
correctly treats source drift as a reason to rebuild.

## Promotion is separate

Retain the declaration with the qualification and review source differences
alongside evidence. Do not relabel a candidate as reference or overwrite the
frozen video manifest. A new default needs a separately reviewed build/profile
revision and the usual promotion gates.
