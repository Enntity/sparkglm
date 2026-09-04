# Results and qualifications

This is the canonical entry point for SparkGLM evidence.

Start with [`CURRENT.md`](CURRENT.md) for the evidence supporting the runnable
engine, then use `index.json` for the complete machine-readable catalog.

- `index.json` lists every checked-in qualification record.
- `accepted/` contains results that passed a declared scoped gate.
- `candidates/` contains work still awaiting promotion evidence.
- `rejected/` preserves measured negative results.
- `legacy/` contains the complete pre-policy experiment archive, migrated
  without pretending it satisfied requirements that did not yet exist.

Read `docs/METHODOLOGY.md` before comparing numbers and
`docs/QUALIFICATION.md` before adding a result. Verify everything with:

```bash
python3 scripts/qualification.py verify-all
```

The prominent result index is intentional: accepted results, limitations, and
negative experiments should be easier to find than an isolated headline.
