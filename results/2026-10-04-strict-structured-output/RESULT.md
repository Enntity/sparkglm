# Strict structured output (engine sparkglm/atlas-20261004-strict)

Before this engine, a `response_format` request (json_schema or json_object) on GLM-5.3 Flash TP2 could return
prose or ```json-fenced text that ignored the schema, plus reasoning text with thinking turned off.

Causes and fixes:
- **First token masked inside `<think>`.** The GLM template opens `<think>` when thinking is off and there are no
  tools, and the first token was masked by the JSON grammar while still inside it. Now response_format renders a
  closed `<think></think>` when thinking is off, and a strict grammar never masks or consumes a thinking token.
- **DFlash verify never masked by the grammar.** On TP2 the verify always takes the raw-argmax path, so drafted
  tokens were never grammar-masked. Now a sequence with a json_schema or json_object grammar decodes serially,
  with every token masked at its own position.
- **Silent fallback to free text.** A grammar refusal used to drop the grammar and continue unconstrained. Now it
  ends the response with an error (finish_reason "error", HTTP 500 when blocking). A schema that cannot be enforced
  returns HTTP 400 with the reason.
- **Tool-turn guards.** Strict JSON output is exempt from the inter-tool prose budget (3,072 tokens) and the
  content-loop watchdog. Tool calls and plain requests behave as before.

Measured on the pair (2026-10-04, production image and profile; `raw/structured.log`):
- strict schema with thinking off: schema-valid JSON and 0 reasoning (×3, plus `reasoning_effort: none`)
- thinking on: reasoning, then schema-valid JSON
- json_object: valid JSON
- a ~28k-character, 36-memory consolidation request: schema-valid JSON 4/4
- requests without a schema: prompt-logprob hash `8c75d2886794` and decode speed unchanged
- tool-call smoke test passes

Cost: schema-constrained requests decode without speculation, at about 13 tok/s.
