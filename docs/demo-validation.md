# Synthetic demo validation

The demo added after v0.1.0 was checked with CUDA disabled, using Python 3.12.14, PyTorch 2.10.0+cu128, Transformers 5.17.0 and the pinned local processor revision. This is a new CPU preparation check, not a new GPU inference result.

Commands from the repository root:

```bash
python make_demo.py
python prepare.py --spec runs/demo/spec.json --output runs/demo-inputs.pt --model /path/to/pinned/local/snapshot
python -m unittest discover -s tests -p test_demo.py -v
```

The processor returned:

| Field | Observed value |
|---|---|
| Model revision | `f75b40e3da2003cdd6e1829b1f420ca70797c34e` |
| `frame_count` | 2 |
| `audio_samples_16khz` | 32000 |
| `audio_valid_frames` | 200 |
| `tokens` | 274 |
| `candidate_token_ids` | [32, 33] |

The decoder tests check PNG checksums and pixel content, PCM duration/rate/amplitude, referenced media paths and refusal to overwrite existing files. Both tests passed. The output tensor and sidecar are local generated files, not historical research evidence.

No model accuracy, GPU memory or stock-attention equivalence result is claimed for this new synthetic input. Use `infer.py --compare-stock` to measure those numerical diagnostics in your own environment; do not confuse them with the archived GPU checks in [release validation](release-validation.md).
