# Contributing

Start with the [quickstart](docs/quickstart.md) and [roadmap](docs/roadmap.md). Documentation, clearer failure messages and small CPU examples are useful contributions; GPU access is not required for those changes.

## Report a problem

Use the [reproduction form](https://github.com/Cookie-27/qwen-omni-16gb/issues/new?template=reproduction.yml). Include the repository commit, exact command, Python version and first failing output. Add PyTorch/Transformers, OS, CUDA and hardware for runtime problems. Prefer bundled or synthetic inputs; do not attach credentials, model weights or private media.

## Make a change

1. Fork the repository and create a branch for one focused improvement.
2. Keep historical evidence immutable. Record new experiments separately with their protocol and provenance.
3. Run the relevant checks below. State checks you could not run.
4. Open a pull request explaining the behavior changed, its purpose and remaining limits. Update both READMEs when a user-facing command changes.

```bash
python verify_evidence.py
python make_demo.py --output-dir runs/contributor-demo
python -m unittest discover -s tests -v
```

Evidence verification and demo generation use only the standard library. Attention tests need PyTorch and Transformers from the pinned runtime; CPU builds are sufficient. The demo output directory must be new.

## Evidence requirements

Claims about accuracy, speed or memory need a matched comparison and an explicit measurement boundary. Preserve input identities, option order, model revision, dtype and readout when claiming numerical equivalence. A failed check should not be fixed by silently loosening its tolerance. Smoke tests demonstrate execution, not task accuracy.

Use the [proposal form](https://github.com/Cookie-27/qwen-omni-16gb/issues/new?template=proposal.yml) for new model/runtime support or costly experiments. State the protocol before inspecting results. Small documentation fixes can go directly to a pull request.

Submissions must be compatible with the repository's [MIT license](LICENSE); third-party model/data terms remain in [NOTICE.md](NOTICE.md).
