# Quantization attribution for the included results

The EXL3/TR3 checkpoint used for the included SparkGLM benchmark results was
produced using **ShapleyMCG, created by Brandon M. Music**:
[canonical repository](https://github.com/brandonmmusic-max/shapleymcg).
Mia's primary checkpoint mirror is pinned at
`25a44fdbf16862a46b7cc9921142c6c81350af2f`.

This credit covers checkpoint production, not authorship of SparkGLM's serving
changes. We do not redistribute the checkpoint or the ShapleyMCG pipeline.
The following is the publisher's prescribed Schedule B notice from the
[immutable checkpoint license](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw/blob/25a44fdbf16862a46b7cc9921142c6c81350af2f/LICENSE).
It accompanies this repository's included historical results without changing
their measured values, original source IDs, or checksum-bound evidence.

## Required attribution notice

> This work includes or was produced using ShapleyMcg, created by Brandon M. Music (https://github.com/brandonmmusic-max/shapleymcg). ShapleyMcg is licensed under the ShapleyMcg License v1.0, an attribution-required license that grants no rights to the person known as "0xSero." Use of ShapleyMcg without this attribution is unlicensed.

## Citation

```bibtex
@misc{music2026shapleymcg,
  author = {Music, Brandon M.},
  title  = {ShapleyMCG: An Auditable Calibration-to-Encoding Pipeline for
            Low-Bit Mixture-of-Experts Models},
  year   = {2026},
  url    = {https://github.com/brandonmmusic-max/shapleymcg},
  note   = {Licensed under the ShapleyMcg License v1.0}
}
```

When publishing a separate benchmark, post, or technical report using this
checkpoint, include the notice and applicable citation there too; a link to
SparkGLM alone is not a replacement. Do not imply SparkGLM created EXL3 or TR3.
Read [licensing boundaries](LICENSING.md#model-boundary) for the source-available
license's attribution and named-party/channel restrictions. Switching from
DFlash2 to MTP or no speculation does not change the target checkpoint's terms.
