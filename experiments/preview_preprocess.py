"""Dump visual antes/depois do preprocess da v2, por imagem.

Gera results/preview/{nome}__0raw.jpg e __1prep.jpg pra inspeção a olho.
Usa exatamente o pipeline da v2 (deskew+crop+clahe). Também imprime o ângulo
de deskew detectado e a mudança de dimensões (indício de crop).
"""

import os
import sys
from pathlib import Path

os.environ["PREPROCESS_VISION"] = "1"
os.environ["PREPROCESS_DESKEW"] = "1"
os.environ["PREPROCESS_CROP_PAPER"] = "1"
os.environ["PREPROCESS_CLAHE"] = "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "system"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from vision.preprocess import preprocess, _decode, _deskew_angle  # noqa: E402

IMAGES = ROOT / "imagens"
PREVIEW = Path(__file__).resolve().parent / "results" / "preview"
PREVIEW.mkdir(parents=True, exist_ok=True)


def dims(b):
    arr = np.frombuffer(b, np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return im.shape[:2] if im is not None else (0, 0)


def main(names):
    for name in names:
        p = IMAGES / name
        if not p.exists():
            print(f"[skip] {name} não existe")
            continue
        raw = p.read_bytes()
        # baseline jpeg (mesmo do harness)
        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        ok, raw_jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        raw_jpg = raw_jpg.tobytes()
        prep = preprocess(raw_jpg)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        angle = _deskew_angle(gray)
        h0, w0 = dims(raw_jpg)
        h1, w1 = dims(prep)

        stem = name.rsplit(".", 1)[0]
        (PREVIEW / f"{stem}__0raw.jpg").write_bytes(raw_jpg)
        (PREVIEW / f"{stem}__1prep.jpg").write_bytes(prep)
        crop = "sim" if (h1, w1) != (h0, w0) else "não"
        print(f"{name:>12}  deskew={angle:+.2f}°  {w0}x{h0} → {w1}x{h1}  crop={crop}")


if __name__ == "__main__":
    args = sys.argv[1:] or ["47.jpeg", "10.jpeg", "100.jpeg", "1.png"]
    main(args)
