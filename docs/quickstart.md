# First input and troubleshooting

## Choose the path you need

| Task | Dependencies | Downloads | GPU |
|---|---|---|---|
| `python verify_evidence.py` | Python 3.11+ standard library | None | No |
| `python make_demo.py` | Python 3.11+ standard library | None | No |
| `python prepare.py ...` | Runtime requirements | Pinned processor files, unless local | No |
| `python infer.py ...` | Runtime requirements and CUDA PyTorch | Pinned model weights, unless local | Yes |

Run commands from the repository root in a dedicated Python environment. Install the CUDA PyTorch build appropriate for your platform before `python -m pip install -r requirements.txt`. CPU tests alone do not demonstrate GPU compatibility. The validated runtime uses PyTorch 2.10.0 and Transformers 5.17.0; the patch deliberately rejects other Transformers versions.

## Generate and prepare a short clip

```bash
python make_demo.py
python prepare.py --spec runs/demo/spec.json --output runs/demo-inputs.pt
```

The generator writes original synthetic media: two 224x224 RGB PNGs (red at 0s, blue at 1s), a two-second 16kHz mono PCM WAV with a 440Hz tone, and a JSON spec. All media paths are relative to that spec. It refuses to overwrite an existing directory. Generated media inherits this repository's MIT license.

Preparation writes `runs/demo-inputs.pt` and `runs/demo-inputs.json`. Inspect `frame_count` (2), `audio_samples_16khz` (32000), `audio_valid_frames` (200), candidate token IDs and tensor hash. It uses the complete audio and both frames. It does not load model weights or call the GPU. Changing the question changes token count; it is not a constant API contract.

To reuse a local copy of the pinned checkpoint, add `--model /path/to/local/snapshot` to both preparation and inference. The path must contain the documented checkpoint/processor revision. The default inference download can include upstream speech components although only the Thinker is loaded; reserve disk capacity for the upstream repository as well as tensor outputs.

## Compare the two attention paths

```bash
python infer.py --inputs runs/demo-inputs.pt --compare-stock --output runs/demo-result.json
```

This runs the short candidate-scoring input through stock attention followed by partitioned attention. Read `comparison.historical_operational_gate_passed` in the output JSON. The fixed gate requires maximum full-vocabulary logit difference ≤ 0.5, candidate-logit difference ≤ 0.125, candidate-probability difference ≤ 0.02, and the same candidate argmax. `comparison.full_vocab_exact` is a separate, stricter diagnostic. A failed gate produces a nonzero exit after saving the result; other runtime errors can fail earlier. Do not change tolerances after observing a failure.

Candidate probabilities are normalized over the listed answer letters. They are not calibrated confidence or unrestricted token probability. Candidate vocabulary mass is reported separately. Synthetic answer A describes the generated content, but there is no promised model answer or fixed probability. This example is a wiring check, not a multimodal benchmark.

For real inputs, copy the generated spec and replace its question, choices, WAV and frames. Extract frames at 1fps; timestamps must be chronological, finite and nonnegative. The prompt assumes 1fps. Every provided frame is processed; there is no MP4 extraction or automatic selection in this repository.

## Common failures

| Symptom | Next step |
|---|---|
| Output directory already exists | Generate into a new `--output-dir`; existing media is preserved. |
| CUDA unavailable | Confirm the installed PyTorch build supports your GPU. Verification and demo generation remain CPU-only. |
| Unsupported Transformers version | Use the pinned version in a separate environment; new-version support needs numerical validation. |
| Model download/access/disk error | Check upstream access and space, or supply an existing pinned snapshot. |
| Stock comparison runs out of memory | Start with the two-second example; long stock inputs historically OOM. A failed stock run is not an equivalence result. |
| Numerical comparison fails | Keep the input and report versions, device, dtype, command and all deltas; do not loosen the gates. |
| Tensor hash differs | Rerun preparation after an intentional media/spec change and keep its tensor and JSON sidecar together. |

The evidence verifier rechecks historical results. It does not execute this new synthetic model input. The new demo is covered by media decode tests and a [real processor check](demo-validation.md); existing GPU validations are described separately in [release validation](release-validation.md).
