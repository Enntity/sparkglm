# RigMark cell set

The RigMark numbers in the result bundle use the 8 × 128K profile
(`PROFILE=8x128k` in `.env`, or `install/profiles/8x128k.json`) and the RiNGSiDE TP2 cell set, thinking on with
`reasoning_effort: low`, two runs per cell.

```sh
git clone https://github.com/othexmr/rigmark.git && cd rigmark
git checkout 40fabcaf6e96                       # the published TP2 comparison commit
cp metadata.example.json metadata.json          # describe your hardware
./rigmark run --base-url http://127.0.0.1:8893 --model auto --metadata metadata.json \
  --comparison-id atlas-vs-ringside-tp2 --runs 2 --prefill-runs 2 --concurrency 1,2,3,4,5,6,8 \
  --concurrency-runs 2 --extra-body '{"chat_template_kwargs":{"reasoning_effort":"low"}}' \
  --label atlas-code-full --prefill-depths 8192,32768,65536 --output rig-code-full.json
./rigmark run --base-url http://127.0.0.1:8893 --model auto --metadata metadata.json \
  --comparison-id atlas-vs-ringside-tp2 --runs 2 --prefill-runs 2 --concurrency 1,2,3,4,5,6,8 \
  --concurrency-runs 2 --extra-body '{"chat_template_kwargs":{"reasoning_effort":"low"}}' \
  --label atlas-prose-concurrency --concurrency-workload prose --skip-prefill \
  --staggered 2,4,6,8 --staggered-runs 2 --staggered-depth 32768 --staggered-incumbent-tokens 1024 \
  --staggered-arrival-tokens 256 --staggered-delay 1.0 --staggered-workload prose \
  --output rig-prose.json
```

The whole set takes about an hour.
