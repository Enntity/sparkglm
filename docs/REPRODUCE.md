# Reproduce our numbers

## Current suite

Run SparkModelBench from rank zero with the server up and the pair dedicated to
the comparison. The Git submodule pins the shared runner and both upstream
RigMark suites. Prepare the CPU quality judge before measuring.

```sh
git submodule update --init --recursive -- bench/sparkmodelbench
docker build -t rigmark-quality-runtime:0.4 bench/sparkmodelbench/upstream/rigmark-quality/quality_suite
cp bench/sparkmodelbench/upstream/rigmark/metadata.example.json /tmp/sparkglm-appliance.json
# Fill in the hardware, model, image and settings of the serving instance.
bench/run.sh --metadata /tmp/sparkglm-appliance.json --out /tmp/sparkglm-benchmark
```

Use a new result directory for each run. The prefill sweep stops at 256K.
Identical comparison IDs, request settings and trial
salts reproduce inputs across contenders. Every result includes a card, raw
JSON and SHA256SUMS. See the [suite protocol](../bench/sparkmodelbench) for how
to interpret output gates, cache evidence and task budgets.

For a product release, run all four lanes and save the reviewed [versioned scorecard](VERSIONING.md#benchmark-each-release).

Earlier studies used the [archived SparkGLM drivers and commands](../bench/archive/README.md).
