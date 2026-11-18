# Results

- [SparkGLM 0.3.6 promotion checks, 2026-11-18](2026-11-18-0.3.6-promotion/RESULT.md): installed through LLooM's recipe from the published image. Strict, image and video output identical to 0.3.5 on a pair; on four Sparks output identical to 0.3.5, 107/109, speed level.
- [SparkGLM 0.3.6, 2026-11-18](2026-11-18-0.3.6/RESULT.md): one engine fix against 0.3.5: a forced tool call opens with the native token. GLM-5.3-Flash output identical, 109/109; Sage S2.1 Flash 109/109 where it passed 97.
- [SparkGLM 0.3.5 promotion checks, 2026-11-16](2026-11-16-0.3.5-promotion/RESULT.md): installed through LLooM's recipe. Strict, image and video output identical to 0.3.3 on a pair; on four Sparks output identical to 0.3.0, 107/109, speed level.
- [SparkGLM 0.3.5, 2026-11-15](2026-11-15-0.3.5/RESULT.md): prompts prefill in arrival order. The first of four cold 16K-token prompts gets its first token in 5.9 s against 11.2 s; the prefill yield is measured and left off.
- [SparkGLM 0.3.5, 2026-11-14](2026-11-14-0.3.5/RESULT.md): concurrent streams on a pair against 0.3.4. Four JSON-schema streams 68.5 tok/s combined against 43.1, with the same tokens.
- [SparkGLM 0.3.4, 2026-11-14](2026-11-14-0.3.4/RESULT.md): in Spark mode a request that turns thinking off reasons at Low effort under a cap: 19–27 s becomes about 3 s. Other output identical to 0.3.3.
- [SparkGLM 0.3.3, 2026-11-11](2026-11-11-0.3.3/RESULT.md): three exact engine gains and the Apache-2.0 reference drafter revision against 0.3.2 on one pair. Output identical in Spark and Strict, 109/109; prose +9.7%, cold 16K prefill +4.1%, KV pool 2.42–2.67M tokens.
- [SparkGLM 0.2.0 on engine c87f2790, 2026-10-27](2026-10-27-engine-c87f2790/RESULT.md): exact-image TP2 qualification of the current pin, which adds Enntity/atlas#16's forced tool_choice and streamed-markup fixes. 109/109 in Strict, Strict native MTP and Spark.
- [SparkGLM 0.2.0 on the common engine, 2026-10-27](2026-10-27-common-engine-020/RESULT.md): exact-image TP2 qualification of Strict, Strict native MTP and Spark with the tool-call guard, on the earlier pin `d0cead14`.
- [Unified TP2/TP4 release, 2026-10-21](2026-10-21-unified-release/RESULT.md): published identity, deployment, current quick decode and provenance.
- [Unified qualification](2026-10-21-unified-rc/RESULT.md): TP2/TP4 Strict/Spark, chain/ring, cache restore and native MTP.
- [Retained TP2 campaign](2026-10-20-tp2-qualification/RESULT.md): full measured suite and its original failed gates, resolved by the later focused checks.


Each directory is one measured configuration: `RESULT.md` says what was run
and what it showed, `raw/` holds the unedited receipts, and `SHA256SUMS`
covers every raw file.

```sh
cd results/2026-09-29-atlas-merged && sha256sum -c SHA256SUMS
```

To add a result, record the image tag (`./start.sh status`), the profile, and
every repetition you ran, including slow ones. Measure the baseline on the
same pair, and put its receipts in `raw/` too.

Results from the vLLM era are preserved at the tag `vllm-final`.
