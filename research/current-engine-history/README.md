# Current engine history

The public release candidate intentionally starts from a clean history so
checkpoint-derived tensors and machine-local artifacts from earlier upstream
snapshots cannot remain reachable through old git objects.

This directory preserves the accepted SparkGLM changes as a mailbox series
from Mia base `eb0469fbb2b49fd7c025f594a3339a121e58f7a9` through SparkGLM
candidate `c80531867e13085c356ae5b9bff4c3b98ee64e8b`. Commit authors, messages,
co-author trailers, and per-change provenance remain inspectable without
publishing the contaminated historical object graph.

The repository root already contains the resulting source tree. These patches
are provenance records, not an additional installation path.
