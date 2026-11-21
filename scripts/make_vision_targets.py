"""Generate the picture targets for smabo-brain's vision modes.

    python scripts/make_vision_targets.py [-o smabo_sim/worlds/targets]

Writes PNGs that a world's ``targets`` can show (``image: targets/<file>``):

  aruco_4x4_50_<id>.png   ArUco markers (DICT_4X4_50, the brain's default), ids 0-3
  qr_smabo.png            a QR code reading "smabo"
  face.png                a drawn face that OpenCV's Haar frontal-face cascade finds

Needs OpenCV (``pip install opencv-python-headless``; smabo-brain's venv has it).
The simulator itself only reads the PNGs, so it does not need OpenCV. Markers
get a white quiet zone so the detector finds their outline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "smabo_sim" / "worlds" / "targets"


def with_border(img: np.ndarray, frac: float) -> np.ndarray:
    pad = int(round(img.shape[0] * frac))
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)


def aruco(marker_id: int, size: int = 240) -> np.ndarray:
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    return with_border(cv2.aruco.generateImageMarker(d, marker_id, size), 1 / 6)


def qr(text: str, size: int = 240) -> np.ndarray:
    code = cv2.QRCodeEncoder.create().encode(text)          # 1 px per module, with quiet zone
    return cv2.resize(code, (size, size), interpolation=cv2.INTER_NEAREST)


def face(n: int = 256) -> np.ndarray:
    """A simple drawn face (BGR). Dark brows/eyes over lighter cheeks is what the
    Haar cascade keys on, so it is detected at 60-200 px."""
    img = np.full((n, n, 3), (200, 205, 210), np.uint8)
    cv2.ellipse(img, (n // 2, n // 2 - n // 6), (int(n * .34), int(n * .30)), 0, 180, 360, (40, 45, 60), -1)  # hair
    cv2.ellipse(img, (n // 2, n // 2 + 6), (int(n * .30), int(n * .40)), 0, 0, 360, (150, 175, 215), -1)      # skin
    ey = int(n * .43)
    for dx in (-1, 1):
        ex = n // 2 + dx * int(n * .12)
        cv2.ellipse(img, (ex, ey - int(n * .07)), (int(n * .075), int(n * .018)), 0, 0, 360, (50, 55, 70), -1)
        cv2.ellipse(img, (ex, ey), (int(n * .06), int(n * .03)), 0, 0, 360, (245, 245, 245), -1)
        cv2.circle(img, (ex, ey), int(n * .028), (40, 35, 30), -1)
        cv2.ellipse(img, (ex, ey + int(n * .035)), (int(n * .06), int(n * .02)), 0, 0, 180, (120, 140, 180), -1)
    cv2.ellipse(img, (n // 2, int(n * .60)), (int(n * .035), int(n * .06)), 0, 0, 360, (125, 150, 195), -1)   # nose
    cv2.ellipse(img, (n // 2, int(n * .70)), (int(n * .09), int(n * .025)), 0, 0, 360, (80, 80, 160), -1)     # mouth
    return cv2.GaussianBlur(img, (0, 0), n / 180)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    files = {f"aruco_4x4_50_{i}.png": aruco(i) for i in range(4)}
    files["qr_smabo.png"] = qr("smabo")
    files["face.png"] = face()
    for name, img in files.items():
        cv2.imwrite(str(args.out / name), img)
        print("wrote", args.out / name)


if __name__ == "__main__":
    main()
