# Licensing and attribution rules

SparkGLM is a multi-license source distribution. The root `LICENSE` is the
default for original SparkGLM work; it does not overwrite the license of an
upstream file, an archive, the chat template, or a downloaded model.

The machine-readable source and file-boundary ledger is
`provenance/upstreams.json`. `python3 tests/test_licensing.py` verifies its
source pins, retained license/notice bytes, path rules, and SPDX consistency.

## Practical boundaries

| Material | Governing terms | What we do |
| --- | --- | --- |
| Original SparkGLM integration | Apache-2.0 | Default for new original files |
| Mia recipe files | MIT | Preserve Mia's complete MIT notice; modifications to those recipe files remain MIT unless the ledger says both MIT and Apache apply |
| vLLM source and direct backports | Apache-2.0 | Preserve vLLM notices and mark modified source/patches |
| ExLlamaV3-derived arithmetic | MIT plus the license of SparkGLM's modifications | Retain the ExLlamaV3 MIT notice and state exactly what was adapted |
| Reederey M64 pipeline | Apache-2.0 plus inherited Mia MIT material | Retain both the Apache license and Reederey/Mia notice |
| Z.AI chat template | GLM-5.3 License | Keep its dedicated license and provenance sidecar |
| Atlas-native archive | AGPL-3.0-only | Keep it under `research/atlas/`; do not copy it into the Apache serving path |
| FlashKDA source | MIT | The external source is not vendored; its license is retained |
| DeepGEMM FP16 integration | MIT and Apache-2.0 | Preserve DeepSeek's MIT notice for patched source and vLLM's Apache notice for the integration |
| FlashKDA slot patch | MIT and AGPL-3.0-only | It contains MIT-derived patch context plus AGPL SparkGLM changes |
| Patch mailboxes | Per target file | A mailbox is an archive, not a relicensing mechanism |

The `.github/FUNDING.yml` sponsor link is intentionally retained from the Mia
recipe and directs sponsorship to MiaAI-Lab.

## Model boundary

No model, quantized weight, drafter weight, or abliteration tensor is licensed
by the root Apache license, and none is committed here. The launcher downloads
immutable revisions from their publishers:

- the primary and fallback EXL3/TR3 checkpoints use the ShapleyMCG License 1.0;
- the default DFlash2 drafter is **CC BY-NC-ND 4.0**, including its
  non-commercial and no-derivatives restrictions;
- the underlying GLM model uses the GLM-5.3 License.

The default matches the published-video configuration; it does not imply that
the drafter's terms fit every operator. Select
`SPEC_METHOD=mtp` or `SPEC_METHOD=none` instead of DFlash2 when its terms do
not. SparkGLM does not copy DFlash2 weights or claim that Apache-2.0 changes
their license.

Container base images and packages installed during a build retain their own
licenses. Publishing a SparkGLM image requires preserving the notices and
source obligations of the image contents, not merely this repository.

## Contributor checklist

1. Classify the change as original, copied, adapted, ported, or inspired.
2. For external work, record the canonical URL, full immutable revision,
   relationship, applicable license, and affected paths in the provenance
   ledger and `docs/ATTRIBUTION.md`.
3. Preserve upstream headers, copyright statements, license texts, NOTICE
   files, and prominent modification notices. Never replace them with the
   root license.
4. Do not use `Co-authored-by` as a thank-you. Use it only when that person
   actually authored code or text in the commit; otherwise use `Provenance:`.
5. Do not copy source or modified artifacts from a no-derivatives checkpoint.
6. Keep model-derived tensors, weights, binaries, credentials, and local paths
   out of Git.
7. Run `./scripts/check.sh all`; then inspect the attribution diff manually.

This is the repository's engineering policy, not legal advice. A public binary
or hosted commercial service may need an additional legal review of all
downloaded and containerized components.
