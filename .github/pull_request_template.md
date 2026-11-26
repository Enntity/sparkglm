## What changes

## Evidence

- [ ] Checks pass (`python3 -m unittest discover -s install -p 'test_*.py'`)
- [ ] For a change to `install/` or the engine pin: `./start.sh` from a clean
      clone on two Sparks, smoke test answered
- [ ] For a version bump, link the candidate’s [SparkModelBench scorecard](../docs/VERSIONING.md#benchmark-each-release) and its checksums.
- [ ] For a speed comparison, measure the baseline on the same pair.

Known regressions, noise or untested boundaries:
