"""Foveate a driver's view, as the vision module sees it.

    python examples/foveated_view.py IMAGE --fov 90 --gaze 320 150 --out foveated.png

The input is any RGB image with a known horizontal field of view, e.g. the
CARLA driver-eye camera from embodied-ui (``evidence/carla-fingertip.png``).
The output shows the original and the foveated percept side by side.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image

from embodied_human.vision.foveation import FoveatedVision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--fov", type=float, default=90.0, help="horizontal field of view, degrees"
    )
    parser.add_argument(
        "--gaze", type=float, nargs=2, help="gaze point in pixels (x y)"
    )
    parser.add_argument("--out", type=Path, default=Path("foveated.png"))
    args = parser.parse_args()
    image = np.asarray(Image.open(args.image).convert("RGB"))
    height, width = image.shape[:2]
    # Pixels per degree at the image centre of a pinhole camera.
    focal = (width / 2) / math.tan(math.radians(args.fov / 2))
    ppd = focal * math.tan(math.radians(1.0))
    gaze = tuple(args.gaze) if args.gaze else (width / 2, height / 2)
    percept = FoveatedVision().perceive(image, gaze, ppd)
    assert percept.image is not None
    out = (np.clip(percept.image, 0, 1) * 255).astype(np.uint8)
    y, x = int(gaze[1]), int(gaze[0])
    out[max(0, y - 1) : y + 2, max(0, x - 6) : x + 7] = (255, 60, 60)
    out[max(0, y - 6) : y + 7, max(0, x - 1) : x + 2] = (255, 60, 60)
    Image.fromarray(np.concatenate([image, out], axis=1)).save(args.out)
    print(
        f"{width}x{height} px, {ppd:.1f} px/deg at the centre, "
        f"gaze {gaze}; wrote {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
