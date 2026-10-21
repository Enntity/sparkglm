# Results

- [Unified TP2/TP4 release, 2026-10-21](2026-10-21-unified-release/RESULT.md): published identity, deployment, current quick decode and provenance.
- [Unified qualification](2026-10-21-unified-rc/RESULT.md): TP2/TP4 Strict/Spark, chain/ring, cache restore and native MTP.
- [Retained TP2 campaign](2026-10-20-tp2-qualification/RESULT.md): full measured suite and its original failed gates, resolved by the later focused checks.


Each directory is one measured configuration: `RESULT.md` says what was run
and what it showed, `raw/` holds the unedited receipts, and `SHA256SUMS`
covers every raw file.

```sh
cd results/2026-09-29-atlas-merged && sha256sum -c SHA256SUMS
```

To add a result, record the image tag (`./start.sh status`), the profile, and
every repetition you ran, including slow ones. Measure the baseline on the
same pair, and put its receipts in `raw/` too.

Results from the vLLM era are preserved at the tag `vllm-final`.
