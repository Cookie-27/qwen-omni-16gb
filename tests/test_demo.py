import json
from pathlib import Path
import struct
import tempfile
import unittest
import wave
import zlib

from make_demo import make_demo


def decode_rgb_png(path):
    """Independently check PNG framing, CRCs and decoded RGB scanlines."""
    data = path.read_bytes()
    if data[:8] != b'\x89PNG\r\n\x1a\n':
        raise AssertionError('Invalid PNG signature')
    position, compressed, header = 8, b'', None
    while position < len(data):
        size = struct.unpack_from('>I', data, position)[0]
        kind = data[position + 4:position + 8]
        payload = data[position + 8:position + 8 + size]
        crc = struct.unpack_from('>I', data, position + 8 + size)[0]
        if crc != zlib.crc32(kind + payload) & 0xffffffff:
            raise AssertionError('Invalid PNG CRC')
        if kind == b'IHDR':
            header = struct.unpack('>IIBBBBB', payload)
        if kind == b'IDAT':
            compressed += payload
        position += 12 + size
        if kind == b'IEND':
            break
    return header, zlib.decompress(compressed)


class DemoTests(unittest.TestCase):
    def test_spec_references_complete_decodable_media(self):
        with tempfile.TemporaryDirectory() as temp:
            source = make_demo(Path(temp) / 'demo')
            spec = json.loads(source.read_text(encoding='utf-8'))
            self.assertEqual([frame['second'] for frame in spec['frames']], [0, 1])
            self.assertEqual(len(set(spec['choices'])), 2)
            with wave.open(str(source.parent / spec['audio'])) as wav:
                self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()),
                                 (1, 2, 16000, 32000))
                samples = struct.unpack('<32000h', wav.readframes(wav.getnframes()))
                self.assertGreater(max(samples), 4000)
                self.assertLess(min(samples), -4000)
                self.assertAlmostEqual(sum(x*x for x in samples) / len(samples), 4096**2 / 2, delta=1000)
            for frame, color in zip(spec['frames'], [(255, 0, 0), (0, 0, 255)]):
                header, decoded = decode_rgb_png(source.parent / frame['path'])
                self.assertEqual(header, (224, 224, 8, 2, 0, 0, 0))
                self.assertEqual(decoded, (b'\x00' + bytes(color) * 224) * 224)

    def test_existing_directory_is_not_modified(self):
        with tempfile.TemporaryDirectory() as temp:
            source = make_demo(Path(temp) / 'demo')
            before = {p.name: p.read_bytes() for p in source.parent.iterdir()}
            with self.assertRaises(FileExistsError):
                make_demo(source.parent)
            self.assertEqual(before, {p.name: p.read_bytes() for p in source.parent.iterdir()})


if __name__ == '__main__':
    unittest.main()
