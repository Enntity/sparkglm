# Native NVFP4 context-capacity experiment

Original synthetic workload for the selected native W4A4/MXFP8 DFlash2 TP2
image. No operator or quantization changes. Context and per-rank cache budget
are varied through LLooM's existing managed-profile controls.

`context_probe.py` runs on the Linux gateway host. It reads the running LLooM
service's API key into process memory without printing or persisting it. The
OpenAI request uses the authenticated gateway; tokenizer sizing and cache
metrics use the private loopback model server. It generates unique synthetic
archive records, sizes the rendered prompt through `/tokenize`, and retrieves
separate START/MIDDLE/END codes through three consecutive tool turns. It records
usage, first tool/content time, completion, exact tool arguments, errors, prompt
hash, and cache metrics. No entity history is used.

Example, after managed startup and readiness:

```sh
python3 context_probe.py --tokens 1047552 --tool-choice auto --output /tmp/context-1m.json
```

The 1024-token reserve accommodates output and follow-up turns within a
1048576-token model limit. Always check actual usage, not only requested size.
A successful tiny response or startup allocation is not long-context proof.
The named-tool mode preserves the actual finish reason separately from exact
argument correctness: this server can emit a valid forced tool call with
`finish_reason: stop`. The first 262K receipt predates that scoring separation;
its false aggregate boolean must not be presented as an incorrect archive code.

This is a capacity/cache and bounded retrieval screen, not broad quality,
long-context concurrency, endurance, or release qualification. Preserve cold
runs, failures, and warm turns separately. Check gateway HTTP transport deadlines
as well as model cache admission: an overall backend deadline alone does not
replace the transport's header/body idle deadlines.

`--replay-request /tmp/context-1m-request.json` reuses a saved request after
checking it against the prior receipt's prompt hash and exact token count. It
is appropriate after a model restart emptied the cache, with unchanged
model/tokenizer/template settings. It does not turn a warm cache into a cold
one. Metrics failures are retained as diagnostic comments so loss of the model
endpoint cannot discard the request's error receipt.
