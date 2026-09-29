# Qwen Omni on 16GB

[中文说明](README.zh-CN.md) · [Measured evidence](docs/evidence.md) · [Limitations](docs/limitations.md)

A small, **inference-only** example for scoring answer candidates with **Qwen2.5-Omni-3B Thinker** on a 16GB RTX 5080.
It partitions **text attention heads** into smaller SDPA calls while preserving each head's complete sequence and corresponding KV head.

**Measured historical run:** full 39.7–60 second audio plus all selected 1fps frames, up to **5,521 input tokens**, **10.202 GiB peak allocated** and **12,001 MiB sampled whole-card usage**. Those values describe this workload, not every 16GB card or arbitrary video length.

This repository scores candidate letters with one forward pass. It does **not** generate speech, stream audio, or claim that unrestricted long-form generation or training fits in 16GB. The published “3B” checkpoint's loaded Thinker has **4,703,464,448 parameters**.

## Verify the evidence without a GPU

```bash
git clone https://github.com/Cookie-27/qwen-omni-16gb.git
cd qwen-omni-16gb
python verify_evidence.py
```

Python 3.11+ and its standard library are sufficient. The command checks evidence hashes, four complete-vocabulary vector pairs, 256 historical predictions and five repeated forwards. No model, media download, or network is needed.

## Run on your own audio and sampled frames

Install a CUDA-compatible PyTorch 2.10.0 build for your platform, then:

```bash
pip install -r requirements.txt
python prepare.py --spec example.json --output runs/inputs.pt
python infer.py --inputs runs/inputs.pt --output runs/result.json
```

`example.json` refers to media you provide; frame times are explicit and chronological:

```json
{
  "audio": "clip.wav",
  "frames": [
    {"path": "frames/000.png", "second": 0},
    {"path": "frames/001.png", "second": 1}
  ],
  "question": "Which instrument is sounding?",
  "choices": ["violin", "piano"]
}
```

Paths are relative to the JSON file. Prepare the video frames at 1fps yourself; the script processes every supplied frame and the complete WAV, resamples audio to 16kHz mono, checks feature length, and never silently clips audio. It does not extract frames from MP4. The model may still fail to interpret the media correctly.

The official checkpoint revision is pinned. Use `--model /path/to/local/snapshot` on both commands to reuse an existing download. By default, fetching the upstream repository may also download speech components even though only the Thinker is loaded; allow disk space accordingly.

## Numerical check on a short input

```bash
python infer.py --inputs runs/short.pt --compare-stock --output runs/comparison.json
python -m unittest discover -s tests -v
```

`--compare-stock` runs stock and partitioned SDPA in one process and reports complete-vocabulary exactness and predeclared tolerances. Use a short input that fits stock attention. Do not relax tolerances after a failing comparison. The historical four short vectors matched exactly; this does not prove equality on all hardware or long inputs.

## How the patch works

For each contiguous group of query heads, select the associated KV head and call the existing Transformers SDPA implementation. Query/key sequence lengths, scaling, masks and causal arguments stay intact. Audio and vision attention continue to use stock SDPA. `heads_per_call=2` was used in the historical experiment.

The context manager restores the previous attention registry even on exceptions. The registry is process-global: use one model at a time, no concurrent model calls or nested patches. Training, dropout, unsupported mask layouts and unvalidated Transformers versions are rejected.

PyTorch already has optimized SDPA backends. This is a bounded workaround for a workload whose stock runtime exhausted memory; it is not a replacement for FlashAttention or a claim of a new algorithm. Prefer an efficient supported backend when it works for your environment.

## What is included?

- `omni_headsplit.py`: the small attention adapter.
- `prepare.py` / `infer.py`: portable media preparation and last-token scoring.
- `tests/`: 32 FP32/BF16 grouped/multi-head attention cases against stock and independent FP64 calculations, plus rejection/restoration tests.
- `evidence/`: exported numeric results and complete short-input vectors. No third-party media or model weights.

The historical run used Windows, PyTorch 2.10.0+cu128 and Transformers 5.17.0. CPU operator checks and evidence reproduction can run independently of CUDA. See [release validation](docs/release-validation.md) for what was actually tested after packaging.

Companion study: [Small VLM Decision Lab](https://github.com/Cookie-27/small-vlm-decision-lab).
Licensing and upstream attribution are in [NOTICE.md](NOTICE.md).
