"""Create a two-second synthetic WAV, two PNG frames and a ready-to-use input spec.

Uses only Python's standard library. No model calls or third-party media.
"""
import argparse
import json
import math
from pathlib import Path
import struct
import wave
import zlib


def solid_png(color, size=224):
    def chunk(kind, data):
        return (struct.pack('>I', len(data)) + kind + data
                + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff))
    header = struct.pack('>IIBBBBB', size, size, 8, 2, 0, 0, 0)
    rows = (b'\x00' + bytes(color) * size) * size
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', header)
            + chunk(b'IDAT', zlib.compress(rows)) + chunk(b'IEND', b''))


def make_demo(output_dir):
    output_dir = Path(output_dir)
    # Requiring a new directory prevents accidental replacement of user media.
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / 'red.png').write_bytes(solid_png((255, 0, 0)))
    (output_dir / 'blue.png').write_bytes(solid_png((0, 0, 255)))
    rate, seconds = 16000, 2
    pcm = b''.join(struct.pack('<h', round(4096 * math.sin(2 * math.pi * 440 * i / rate)))
                   for i in range(rate * seconds))
    with wave.open(str(output_dir / 'tone.wav'), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    spec = {
        'audio': 'tone.wav',
        'frames': [{'path': 'red.png', 'second': 0}, {'path': 'blue.png', 'second': 1}],
        'question': 'Which description matches the frames and the audio?',
        'choices': ['A red frame followed by a blue frame, with a steady tone.',
                    'A blue frame followed by a red frame, with silence.'],
    }
    (output_dir / 'spec.json').write_text(json.dumps(spec, indent=2) + '\n', encoding='utf-8', newline='\n')
    return output_dir / 'spec.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('runs/demo'))
    args = parser.parse_args()
    try:
        spec = make_demo(args.output_dir)
    except FileExistsError:
        parser.error('Output directory already exists; choose a new --output-dir.')
    print(f'Created input spec: {spec.as_posix()}')
    print('Media: 2.0 seconds of 16kHz mono audio; two 224x224 frames at 0s and 1s.')
    print('Synthetic pipeline check only; no inference or accuracy evaluation performed.')


if __name__ == '__main__':
    main()
