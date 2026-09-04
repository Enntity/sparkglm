# GLM-5.3 chat-template provenance

`chat_template.jinja` is an adaptation of Z.AI's GLM-5.3 chat template at
commit `aca966e4e02791568aa6a4ced368624b3d897f42`:

https://huggingface.co/zai-org/GLM-5.3/blob/aca966e4e02791568aa6a4ced368624b3d897f42/chat_template.jinja

The pinned upstream file has SHA-256
`3740abcea51c45830cb3ca562084ad5fb2ef53589376f73332e9886f93ade41c`.

This copy imports the upstream `None` guard and the four early exits added to
the tool-result reordering validation. It retains the serving recipe's prior
extensions for `thinking` / `enable_thinking`, prefix-stable thinking-off
rendering, and GLM multimodal sentinel tokens. Those local extensions are why
the bundled file is not expected to be byte-identical to upstream.

The upstream file is provided under the GLM-5.3 License. Its full license and
copyright notice are retained at `LICENSES/GLM-5.3.txt`.
