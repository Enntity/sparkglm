# Contributing

Small, focused changes are welcome: fixes, documentation, launch reliability
and measured improvements. You don't need Sparks for most of them.

Run the same checks as CI before opening a pull request:

```sh
python3 -m unittest discover -s install -p 'test_*.py'
python3 -m unittest discover -s install/converter
shellcheck -S warning -e SC2054 start.sh install/*.sh install/flash_kda/rebuild.sh
python3 -m py_compile bench/*.py
```

**Changes to `install/`** change the image. Before merging, test them with
`BUILD=1 ./start.sh` from a clean clone on two Sparks.

**Performance changes** need raw receipts under `results/`, with the baseline
run on the same pair (see [results/README.md](results/README.md)).

**Engine changes** go to [Enntity/atlas](https://github.com/Enntity/atlas)
first; here you then update the pin in `install/atlas-source.json`.

Credit other people's work with a `Provenance:` line naming the source and
its license. Use `Co-authored-by` only for someone who actually wrote part of
the commit.
