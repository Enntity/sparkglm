# Results

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
