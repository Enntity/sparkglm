# Earlier adaptive staggered prose captures

Four requests arrived1s apart, each about16K input tokens and exactly400 output tokens. SparkGLM adaptive pair times were68.64,71.22,70.51s (median70.51s); the earlier Mia-based EXL3 candidate times were114.93,102.15,102.40s (median102.40s). SparkGLM median completion was31.1% shorter.

This Mia arm was the earlier experimental combined recipe, with FAST/BF16 large-M KDA/denseFP8/fair-prefill options, **not the stock Mia recipe used in the coding comparison**. The captures were sequential, not interleaved. Input counts differed by seven tokens due to templates. Do not combine these observations into a universal stock-Mia ranking or an isolated adaptive-on/off effect. Fresh cache namespaces prevented cross-run reuse; SparkGLM had no foreign local traffic or preemptions in its retained captures. An attribution-lookup issue in the discarded warmup was resolved by shorter caller tags; it was not substituted for a retained sample.

[EXL3 quant attribution and notice](../../../docs/QUANT_ATTRIBUTION.md). Raw recordings remain private.
