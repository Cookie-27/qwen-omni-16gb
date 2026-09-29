"""One-forward, last-token candidate scoring; no speech or text generation."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time
from prepare import MODEL_ID, MODEL_REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=MODEL_ID)
    parser.add_argument('--inputs', type=Path, required=True, help='Tensor pack from prepare.py, with sibling .json metadata')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--heads-per-call', type=int, default=2)
    parser.add_argument('--memory-fraction', type=float, default=.8)
    parser.add_argument('--compare-stock', action='store_true', help='Compare on a SHORT input that fits stock SDPA')
    parser.add_argument('--stock', action='store_true', help='Use only stock SDPA')
    args = parser.parse_args()
    if args.stock and args.compare_stock:
        parser.error('--stock and --compare-stock are mutually exclusive')
    if not 0 < args.memory_fraction <= 1 or args.heads_per_call < 1:
        parser.error('Invalid resource limits')
    meta = json.loads(args.inputs.with_suffix('.json').read_text(encoding='utf-8'))
    if meta.get('model_id') != MODEL_ID or meta.get('model_revision') != MODEL_REVISION:
        raise ValueError('Unexpected processor/model revision')
    if hashlib.sha256(args.inputs.read_bytes()).hexdigest() != meta['tensor_pack_sha256']:
        raise ValueError('Input tensor checksum mismatch')
    import torch
    from transformers import Qwen2_5OmniThinkerConfig, Qwen2_5OmniThinkerForConditionalGeneration
    from omni_headsplit import head_partitioned_sdpa
    torch.set_num_threads(4)
    torch.manual_seed(20260928)
    if not torch.cuda.is_available():
        raise RuntimeError('The inference example requires CUDA; use verify_evidence.py for CPU-only reproduction')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    load_options = {} if Path(args.model).exists() else {'revision': MODEL_REVISION}
    if Path(args.model).exists():
        cfg_path = Path(args.model) / 'config.json'
    else:
        from huggingface_hub import hf_hub_download
        cfg_path = Path(hf_hub_download(args.model, 'config.json', revision=MODEL_REVISION))
    cfg = Qwen2_5OmniThinkerConfig(**json.loads(cfg_path.read_text(encoding='utf-8'))['thinker_config'])
    model, loading = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        args.model, config=cfg, dtype=torch.bfloat16, device_map={'': 'cuda:0'},
        attn_implementation='sdpa', output_loading_info=True, **load_options)
    if any(loading.get(k) for k in ['missing_keys', 'mismatched_keys', 'error_msgs']):
        raise RuntimeError('Incomplete Thinker checkpoint load')
    if not all(k.startswith(('talker.', 'token2wav.')) for k in loading.get('unexpected_keys', [])):
        raise RuntimeError('Unexpected non-speech checkpoint keys')
    params = sum(p.numel() for p in model.parameters())
    if params != 4703464448:
        raise RuntimeError('This example is validated for the documented 3B Thinker checkpoint only')
    model.eval().requires_grad_(False)
    model.lm_head.float()
    handle = model.lm_head.register_forward_pre_hook(lambda module, arguments: (arguments[0][:, -1:, :].float(),))
    cpu_inputs = torch.load(args.inputs, map_location='cpu', weights_only=True)
    if not isinstance(cpu_inputs, dict) or not all(isinstance(v, torch.Tensor) for v in cpu_inputs.values()):
        raise ValueError('Expected a tensor-only input dictionary')
    if cpu_inputs['input_ids'].shape[0] != 1:
        raise ValueError('The example supports batch size one')
    inputs = {k: v.to('cuda', dtype=torch.bfloat16) if v.is_floating_point() else v.to('cuda')
              for k, v in cpu_inputs.items()}
    ids = meta['candidate_token_ids']
    if len(set(ids)) != len(ids) or len(ids) < 2 or any(not isinstance(i, int) or not 0 <= i < cfg.text_config.vocab_size for i in ids):
        raise ValueError('Invalid candidate token IDs')
    results, vectors = {}, {}
    modes = ['stock', 'headsplit'] if args.compare_stock else ['stock' if args.stock else 'headsplit']
    try:
        for mode in modes:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            context = head_partitioned_sdpa(args.heads_per_call) if mode == 'headsplit' else nullcontext({})
            torch.cuda.synchronize()
            start = time.perf_counter()
            with context as stats, torch.inference_mode():
                output = model(**inputs, use_cache=False, use_audio_in_video=False, return_dict=True)
                vector = output.logits[0, -1].float()
                if not torch.isfinite(vector).all():
                    raise RuntimeError('Nonfinite logits')
                candidate = vector[ids]
                probability = candidate.softmax(-1)
                mass = (candidate.logsumexp(0) - vector.logsumexp(0)).exp()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                prediction = int(candidate.argmax())
                results[mode] = {'prediction_index': prediction, 'choice': meta['choices'][prediction],
                                 'candidate_logits': candidate.cpu().tolist(), 'candidate_probabilities': probability.cpu().tolist(),
                                 'candidate_vocab_mass': float(mass), 'forward_seconds': elapsed,
                                 'peak_allocated_GiB': torch.cuda.max_memory_allocated()/2**30,
                                 'peak_reserved_GiB': torch.cuda.max_memory_reserved()/2**30,
                                 'attention_stats': dict(stats)}
                vectors[mode] = vector.cpu().clone()
                del output, vector, candidate, probability, mass
            if mode == 'headsplit' and not stats['text_partitioned_calls']:
                raise RuntimeError('Patch did not intercept text attention')
    finally:
        handle.remove()
    record = {'results': results, 'tokens': int(inputs['input_ids'].shape[-1]), 'thinker_parameters': params,
              'training_updates': 0, 'generation_tokens': 0,
              'timing_scope': 'Forward only; excludes model loading, media preprocessing and input transfer',
              'memory_scope': 'PyTorch allocator peaks, not whole-card sampling'}
    if args.compare_stock:
        a, b = vectors['stock'], vectors['headsplit']
        full = float((a-b).abs().max())
        delta = float((a[ids]-b[ids]).abs().max())
        pdiff = float((a[ids].softmax(-1)-b[ids].softmax(-1)).abs().max())
        passed = full <= .5 and delta <= .125 and pdiff <= .02 and int(a[ids].argmax()) == int(b[ids].argmax())
        record['comparison'] = {'full_vocab_exact': torch.equal(a, b), 'max_full_vocab_abs': full,
                                'max_candidate_logit_abs': delta, 'max_candidate_probability_abs': pdiff,
                                'historical_operational_gate_passed': passed}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record, indent=2))
    if args.compare_stock and not record['comparison']['historical_operational_gate_passed']:
        raise RuntimeError('Predeclared numerical gate failed; do not relax it after observing this run')


if __name__ == '__main__':
    main()
