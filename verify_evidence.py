"""Verify historical numeric evidence with Python's standard library only."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent / 'evidence'


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def main():
    for entry in read('manifest.json')['files']:
        path = (ROOT / entry['path']).resolve()
        if not path.is_relative_to(ROOT.resolve()):
            raise ValueError('Evidence path escapes its directory')
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError(f'Checksum mismatch: {entry["path"]}')
    for index in (0, 67, 132, 255):
        a = (ROOT / 'vectors' / f'stock_{index:03d}.npy').read_bytes()
        b = (ROOT / 'vectors' / f'headsplit_{index:03d}.npy').read_bytes()
        if a != b:
            raise ValueError('A historical full-vocabulary vector pair differs')
    rows = read('predictions.json')
    if len(rows) != 256 or len({r['index'] for r in rows}) != 256:
        raise ValueError('Unexpected historical cohort')
    counts = {m: 0 for m in ('real', 'silent', 'gray', 'both')}
    for row in rows:
        logits = row['candidate_logits']
        e = [math.exp(x-max(logits)) for x in logits]
        probabilities = [x/sum(e) for x in e]
        if any(abs(a-b) > 2e-7 for a, b in zip(probabilities, row['candidate_probs'])):
            raise ValueError('Candidate probabilities do not reproduce')
        index = max(range(len(logits)), key=lambda i: logits[i])
        if index != row['prediction_display_index'] or row['correct'] != (index == row['correct_display_index']):
            raise ValueError('Prediction or label mapping mismatch')
        counts[row['mode']] += row['correct']
    if counts != {'real': 40, 'silent': 41, 'gray': 35, 'both': 32}:
        raise ValueError('Historical counts differ')
    main_rows = {r['index']: r for r in rows}
    replay = read('reloads.json')
    for row in replay:
        for key in ('candidate_logits', 'candidate_probs', 'allowed_vocab_mass', 'full_vocab_argmax', 'prediction_display_index', 'correct'):
            if row[key] != main_rows[row['index']][key]:
                raise ValueError('Historical replay differs')
    print('Verified four byte-identical full-vocabulary vector pairs, 256 predictions and five repeated forwards.')
    print('Historical counts / 64 contexts:', counts)
    print('32 development questions x two orders; chat initialization; zero training. Not a full benchmark.')


if __name__ == '__main__':
    main()
