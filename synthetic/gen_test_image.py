"""
Small synthetic test images for the image round-trip pipeline.

    test_16x16.jpg   16x16 red square on white (JPEG)
    test_32x32.png   32x32 diagonal gradient   (PNG)
    test_8x8.bmp     8x8 black/white checkerboard (BMP)

Pillow is used because it is already installed in the project environment (checked with
`python -c "import PIL; print(PIL.__version__)"`: 12.3.0); nothing is pip-installed.

    python synthetic/gen_test_image.py          # writes synthetic/images/, prints size + magic
"""

import os
import sys
from typing import Dict, List

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(ROOT, "synthetic", "images")


def red_square_16() -> Image.Image:
    """16x16 white image with a red 8x8 square in the middle."""
    im = Image.new("RGB", (16, 16), (255, 255, 255))
    for x in range(4, 12):
        for y in range(4, 12):
            im.putpixel((x, y), (255, 0, 0))
    return im


def gradient_32() -> Image.Image:
    """32x32 diagonal gradient: brightness grows with x + y, colour varies along each axis."""
    im = Image.new("RGB", (32, 32))
    for x in range(32):
        for y in range(32):
            im.putpixel((x, y), (8 * x, 4 * (x + y), 8 * y))
    return im


def checkerboard_8() -> Image.Image:
    """8x8 black/white checkerboard, one pixel per square."""
    im = Image.new("RGB", (8, 8))
    for x in range(8):
        for y in range(8):
            im.putpixel((x, y), (255, 255, 255) if (x + y) % 2 else (0, 0, 0))
    return im


def generate_test_images(out_dir: str = IMAGE_DIR) -> List[Dict[str, object]]:
    """Write the three test images; returns [{path, bytes, magic_hex}] per image."""
    os.makedirs(out_dir, exist_ok=True)
    out = []
    for name, im, kw in (("test_16x16.jpg", red_square_16(), {"quality": 90}),
                         ("test_32x32.png", gradient_32(), {}),
                         ("test_8x8.bmp", checkerboard_8(), {})):
        path = os.path.join(out_dir, name)
        im.save(path, **kw)
        with open(path, "rb") as fh:
            data = fh.read()
        out.append({"path": path, "bytes": len(data), "magic_hex": data[:8].hex(" ").upper()})
    return out


def main() -> int:
    for r in generate_test_images(sys.argv[1] if len(sys.argv) > 1 else IMAGE_DIR):
        print(f"{os.path.basename(r['path']):16s} {r['bytes']:6d} bytes  first 8 bytes: {r['magic_hex']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
