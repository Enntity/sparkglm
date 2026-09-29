## What changes

## Evidence

- [ ] Checks pass (`python3 -m unittest discover -s install -p 'test_*.py'`)
- [ ] For a change to `install/` or the engine pin: `./start.sh` from a clean
      clone on two Sparks, smoke test answered
- [ ] For a performance claim: raw receipts and `SHA256SUMS` under `results/`,
      with the baseline measured on the same pair

Known regressions, noise or untested boundaries:
