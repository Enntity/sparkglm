# Publication review

Release maintainers must complete this review before publishing source or images.

## Prepared for final review

The publication repository is `Enntity/sparkglm`. Original development and
pre-publication histories are preserved separately in private archives. Do not
merge or mirror their Git history into the publication branch. Publication uses
a fresh GitHub repository so old private commits and CI attachments are not
carried into the public repository by a visibility flip.

The current build guide is [SPARKGLM.md](../SPARKGLM.md). The final presentation
cleanup changes no serving engine code or defaults. Contributor tooling now
has a separate declared-hash candidate build and an explicit-image tinyGLM
wrapper regression fix. Detailed build-reproduction
diagnostics remain linked from [current status](../results/CURRENT.md), not
promoted as a separate performance claim.

The publication-history audit removed an upstream machine-local output path
from an old script revision and completed two missing provenance trailers.
The current file tree was unchanged by that cleanup. The
[commit map](../provenance/publication-commit-map.txt) connects original and
sanitized revisions; benchmark receipts keep their original measured IDs and
checksums. The pre-cleanup history remains in the private archive. Historical
contributor names are retained; personal or machine-local mail addresses may be
replaced by the contributor's GitHub noreply address for publication privacy.
The additional privacy cleanup makes archived appliance addresses required
configuration, without changing the current serving source or raw receipts.
Its [second commit map](../provenance/publication-privacy-commit-map.txt) maps
the reviewed revisions to the publication privacy rewrite. Chain it with the
earlier map when tracing an original revision; measured source IDs stay intact.

Source checks, all 52 qualification records, and the existing hosted CI logs
were checked during preparation. These are automated checks, not a substitute
for the human licensing/claims review below or full G3/G4/G5 qualification.
Repository visibility remains a separate approval.

GitHub `main` requires the hosted `quick` check from GitHub Actions, an
up-to-date branch, and resolved review conversations. Force pushes and branch
deletion are disabled; administrators retain recovery bypass. CI uses hosted
CPU runners, not the private Sparks. Private vulnerability reporting could not
be enabled during private preparation (GitHub returned 404); verify and enable
it as part of the visibility-change checklist before directing security
reports there.

The consumer and contributor entry points are now self-contained. Source-only
setup is checked in a fresh Python environment; reference source checks were
also exercised against the retained image without a GPU or model load.
The new candidate builder has source/command-generation tests; it is not a
newly hardware-qualified engine. No new speed or full-model quality claim is
made by this onboarding work.

Tonight's publication target is a **source research preview**, not a named
G5 appliance release. No prebuilt image is offered until the separate
[binary audit](IMAGE_RELEASE.md) and registry-pull qualification are complete.

## Automated gate

- [ ] `./scripts/publication-audit.sh` passes from a clean checkout.
- [ ] All JSON benchmark receipts parse.
- [ ] Shell launch and build scripts pass `bash -n`.
- [ ] `scripts/check.sh all` and qualification verification pass.
- [ ] Root Docker `COPY` inputs are complete and a clean ARM64 build has passed
      on a Spark; static context validation alone is not a built image.
- [ ] No checkpoint, tensor, shared object, cache, credential, or machine-local
      environment file is tracked.
- [ ] No private workstation path, appliance address, or private hostname appears
      in tracked files or reachable history; run
      `python3 scripts/check_publication_privacy.py --history` on a full clone.
- [ ] Confirm the public remote contains only the sanitized root history; do
      not push the source clone's inherited branches or tags.

## Attribution and licensing

- [ ] Review every row in `docs/ATTRIBUTION.md` against the linked source.
- [ ] Confirm `LICENSE`, `NOTICE`, and every file in `LICENSES/` are retained.
- [ ] Confirm the Z.AI chat-template notice and `LICENSES/GLM-5.3.txt` remain
      present with the pinned template provenance.
- [ ] Confirm Atlas engine material remains under `research/atlas/` and labeled
      AGPL-3.0-only; the standalone staggered benchmark and archived campaign
      harnesses are also AGPL and are disclosed separately from serving code.
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
- [ ] Confirm ShapleyMCG's required notice, author, canonical link, and technical
      citation accompany checkpoint results. Disclose source-available and
      named-party/channel restrictions independently of drafter selection.
- [ ] Include the same applicable attribution in separately published benchmark
      posts; repository attribution does not retroactively update other sites.

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
- [ ] Review `docs/KNOWN_LIMITATIONS.md`, especially the 2051-to-2048 sparse-MLA
      candidate-set approximation, and ensure the README disclosure remains.
- [ ] Confirm grouped prefill, cooperative decode, vision, and DFlash2 are
      disclosed as the posted-video defaults, with their legacy evidence,
      incomplete current qualification, rollback controls, and license limits
      visible before performance claims or launch instructions.

## Repository presentation

- [ ] Verify the root README points users to the current vLLM engine first.
- [ ] Verify the default image name is SparkGLM-owned and does not overwrite or
      masquerade as an upstream Mia image tag.
- [ ] Verify the default API bind is loopback and any public/LAN example uses a
      bearer key.
- [ ] Decide whether generated videos belong in a GitHub release; do not commit
      the entire local render archive.
- [ ] Confirm `Enntity/sparkglm` is the reviewed publication repository and
      the private development archive remains separate.
- [ ] Confirm the deliberately retained `.github/FUNDING.yml` MiaAI-Lab
      sponsor link still matches the disclosure in `docs/LICENSING.md`.
- [ ] Verify GitHub description, topics, default branch, issue templates, and
      security policy.
- [ ] Enable the static workflow as a required branch check before accepting
      public pull requests; never attach fork PRs directly to Spark runners.
- [ ] Verify branch protection after the visibility change and enable/test
      private vulnerability reporting and the security-template link.
- [ ] Make the visibility change only as a separate, explicit action after this
      review.
