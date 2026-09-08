# E3 top-8 cardinality follow-up

Six reference-relative correctness cases passed at H4096/I1024, 288 experts,
input tokens 2048/4096/7168 and exactly eight routed rows per input token in
aggregate. Balanced and skewed distributions include zero-count experts. These
synthetic route tables preserve cardinality, not a real router trace.

The initial 2048-token threshold is rejected: balanced and skewed 2048 cases
regressed about 7% and 3%. Balanced 4096 was neutral (+0.5%), skewed 4096 was
42.6% faster; balanced/skewed 7168 were 55.1%/33.2% faster. Seven alternating
timing samples per case are retained. These are fat-path timings; thin experts
and complete model costs are not included. No full-model gain is implied.

The next opt-in policy uses E3 only at input tokens >=4096. Required next gates:
CUDA sanitizer, tinyGLM TP2, repeated full-model performance and quality.
