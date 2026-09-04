# Historical vLLM optimization campaign

This archive predates the clean Mia-derived current branch at the repository
root. It preserves the experimental trail rather than presenting every patch
as useful.

- `../../results/legacy/` contains the indexed accepted, rejected, and raw
  measurements migrated from this campaign.
- `benchmarks/` contains the corresponding public-safe harnesses and raw JSON.
- `docs/` contains the execution analysis and upstream audits.
- `scripts/` contains generic build helpers; machine-local LLooM alias tooling
  was deliberately excluded.
- `patches/` is a mailbox export of the experiment commits from `449b578747`
  through `772c605675`. These patches are historical and depend on the old
  source-fork baseline; they are for inspection, not blind application to the
  current engine.
- `current-followups/` preserves the final rejected fusion variants and route
  profiling tools that had not been committed to the source fork.

The historical source fork started from vLLM
`487ecf187d3dfe74d2cf6119a92881dba403c219`. The maintained and reproducible
serving path is the repository root.
