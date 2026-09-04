# Private publication review

The repository must remain private until a human completes this checklist.

## Automated gate

- [ ] `./scripts/publication-audit.sh` passes from a clean checkout.
- [ ] All JSON benchmark receipts parse.
- [ ] Shell launch and build scripts pass `bash -n`.
- [ ] `scripts/check.sh all` and qualification verification pass.
- [ ] No checkpoint, tensor, shared object, cache, credential, or machine-local
      environment file is tracked.
- [ ] No private workstation path or private Spark hostname appears outside an
      explicitly historical patch-mail header.
- [ ] Confirm the public remote contains only the sanitized root history; do
      not push the source clone's inherited branches or tags.

## Attribution and licensing

- [ ] Review every row in `docs/ATTRIBUTION.md` against the linked source.
- [ ] Confirm `LICENSE`, `NOTICE`, and every file in `LICENSES/` are retained.
- [ ] Confirm the Z.AI chat-template notice and `LICENSES/GLM-5.3.txt` remain
      present with the pinned template provenance.
- [ ] Confirm Atlas-derived material remains confined to `research/atlas/` and
      labeled AGPL-3.0-only.
- [ ] Confirm Mia's MIT notice is preserved and prominent.
- [ ] Confirm Reederey, ExLlamaV3, vLLM PR authors, FlashKDA, model, quant, and
      drafter sources are credited.
- [ ] Confirm no model license is implied to cover code and no code license is
      implied to cover model weights.
- [ ] Run `python3 tests/test_licensing.py` and manually review
      `provenance/upstreams.json` plus every attribution-related diff.
- [ ] Confirm every new runtime commit passes
      `scripts/check_commit_provenance.py` for the proposed public range.
- [ ] Confirm DFlash2's CC BY-NC-ND 4.0 warning and MTP/none alternative are
      visible before the first download/run instructions.

## Technical claims

- [ ] Read `docs/RESULTS.md` beside the primary receipts.
- [ ] Confirm the public headline does not add non-identical A/B percentages.
- [ ] Confirm mixed-scheduler gains are described as queueing/service-policy
      gains rather than equivalent raw GPU acceleration.
- [ ] Confirm cooperative-decode claims retain the tinyGLM/full-model boundary.
- [ ] Confirm Atlas is labeled archival and incomplete, not production-ready.
- [ ] Confirm every public performance claim resolves through `results/index.json`
      to a checksum-valid qualification and its limitations.
- [ ] Confirm legacy evidence is never described as post-policy certification.

## Repository presentation

- [ ] Verify the root README points users to the current vLLM engine first.
- [ ] Decide whether generated videos belong in a GitHub release; do not commit
      the entire local render archive.
- [ ] Decide whether the staging repository should be renamed to `sparkglm`
      before public release.
- [ ] Confirm the deliberately retained `.github/FUNDING.yml` MiaAI-Lab
      sponsor link still matches the disclosure in `docs/LICENSING.md`.
- [ ] Verify GitHub description, topics, default branch, issue templates, and
      security policy.
- [ ] Enable the static workflow as a required branch check before accepting
      public pull requests; never attach fork PRs directly to Spark runners.
- [ ] Make the visibility change only as a separate, explicit action after this
      review.
