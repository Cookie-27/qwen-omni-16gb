"""Prepare a CPU tensor pack from full audio and explicitly sampled image frames."""
import argparse
import hashlib
import json
from pathlib import Path

MODEL_ID = 'Qwen/Qwen2.5-Omni-3B'
MODEL_REVISION = 'f75b40e3da2003cdd6e1829b1f420ca70797c34e'


def prepare(spec_path, output, model=MODEL_ID):
    import math
    import numpy as np
    import torch
    from PIL import Image
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from transformers import Qwen2_5OmniProcessor
    torch.set_num_threads(4)
    spec_path = Path(spec_path).resolve()
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    choices = spec['choices']
    if not 2 <= len(choices) <= 26 or len(set(choices)) != len(choices):
        raise ValueError('Use 2 to 26 distinct choices')
    if not spec['frames']:
        raise ValueError('At least one explicitly sampled frame is required')
    times = [float(frame['second']) for frame in spec['frames']]
    if any(not math.isfinite(t) or t < 0 for t in times) or times != sorted(set(times)):
        raise ValueError('Frame times must be finite, nonnegative, strictly increasing')
    paths = [(spec_path.parent / frame['path']).resolve() for frame in spec['frames']]
    images = []
    for path in paths:
        with Image.open(path) as im:
            images.append(im.convert('RGB'))
    audio_path = (spec_path.parent / spec['audio']).resolve()
    rate, raw = wavfile.read(audio_path)
    if raw.ndim not in (1, 2) or len(raw) == 0 or rate <= 0:
        raise ValueError('Expected a nonempty mono/stereo WAV')
    if np.issubdtype(raw.dtype, np.signedinteger):
        raw = raw.astype(np.float32) / float(-np.iinfo(raw.dtype).min)
    elif raw.dtype == np.uint8:
        raw = (raw.astype(np.float32) - 128) / 128
    elif np.issubdtype(raw.dtype, np.floating):
        raw = raw.astype(np.float32)
    else:
        raise ValueError('Unsupported WAV sample dtype')
    if raw.ndim == 2:
        raw = raw.mean(axis=1)
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite audio samples')
    divisor = math.gcd(rate, 16000)
    audio = resample_poly(raw, 16000 // divisor, rate // divisor).astype(np.float32) if rate != 16000 else raw
    kwargs = {} if Path(model).exists() else {'revision': MODEL_REVISION}
    proc = Qwen2_5OmniProcessor.from_pretrained(model, **kwargs)
    content = [{'type': 'text', 'text': 'Video frames in chronological order, sampled once per second:'}]
    for frame in spec['frames']:
        content.extend([{'type': 'text', 'text': str(frame['second']) + 's:'}, {'type': 'image'}])
    options = '\n'.join(f'{chr(65+i)}: {choice}' for i, choice in enumerate(choices))
    content.extend([{'type': 'text', 'text': 'Complete video audio:'}, {'type': 'audio'},
                    {'type': 'text', 'text': spec['question'] + '\n' + options + '\nSelect exactly one option. Respond only with its letter.'}])
    text = proc.apply_chat_template([
        {'role': 'system', 'content': [{'type': 'text', 'text': 'You are a helpful assistant.'}]},
        {'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
    inputs = dict(proc(text=[text], images=images, audio=[audio], return_tensors='pt',
                       images_kwargs={'min_pixels': 3136, 'max_pixels': 224**2},
                       audio_kwargs={'sampling_rate': 16000, 'padding': 'max_length',
                                     'truncation': False, 'return_attention_mask': True}))
    valid = int(inputs['feature_attention_mask'].sum())
    if valid != int(np.ceil(len(audio) / 160)):
        raise ValueError('Audio feature length differs from the complete input')
    if len(inputs['image_grid_thw']) != len(images):
        raise ValueError('Frame count changed during processing')
    base = proc.tokenizer.encode(text, add_special_tokens=False)
    token_ids = []
    for letter in [chr(65+i) for i in range(len(choices))]:
        ids = proc.tokenizer.encode(letter, add_special_tokens=False)
        if len(ids) != 1 or proc.tokenizer.encode(text + letter, add_special_tokens=False) != base + ids:
            raise ValueError('Answer letter is not a single appendable token')
        token_ids.append(ids[0])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(inputs, output)
    meta = {'schema_version': 1, 'model_id': MODEL_ID, 'model_revision': MODEL_REVISION,
            'candidate_token_ids': token_ids, 'choices': choices, 'frame_count': len(images),
            'audio_samples_16khz': len(audio), 'audio_valid_frames': valid,
            'tokens': int(inputs['input_ids'].shape[-1]),
            'tensor_pack_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
            'source_audio_sha256': hashlib.sha256(audio_path.read_bytes()).hexdigest(),
            'source_frame_sha256': [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]}
    output.with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default=MODEL_ID)
    args = parser.parse_args()
    print(json.dumps(prepare(args.spec, args.output, args.model), indent=2))


if __name__ == '__main__':
    main()
