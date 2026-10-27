"""Pré-processamento clássico (OpenCV) da foto do ECG antes da vision.

Objetivo: reduzir o ruído de captura (foto torta, fundo escuro, contraste
baixo) pra Sonnet ter menos chance de errar medições de ST/QRS por defeito
do meio de entrada — não do raciocínio.

Etapas determinísticas (sem treino, sem dependência de dados):

1. DESKEW — detecta a inclinação dominante do papel via gradiente / Hough e
   roda pra ficar reto. Falha sem rotação se ângulo for pequeno (<0.5°) ou
   ambíguo.
2. CROP DO PAPEL — segmenta o retângulo claro do papel quadriculado contra
   fundo escuro (mesa, mão). Reduz área da imagem que o modelo precisa
   "ignorar".
3. CONTRAST NORMALIZE — CLAHE em canal L do Lab, reforça as linhas do
   traçado sem estourar a grade.

Tudo idempotente e degrada para a imagem original se algo falha (preferimos
imagem ruim a falha de pipeline). Os 3 estágios são plugáveis via flags do
config — fácil de desligar individualmente em A/B.
"""

import io
import sys
from pathlib import Path
from typing import Union

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from loguru import logger

from config import (
    PREPROCESS_VISION,
    PREPROCESS_DESKEW,
    PREPROCESS_CROP_PAPER,
    PREPROCESS_CLAHE,
)

# Limites: ângulos menores que isso não compensam a rotação (introduz blur de
# interpolação). Recortes menores que isso provavelmente são falso-positivos
# (lente que cortou parte do papel) — preserva original.
_MIN_DESKEW_ANGLE = 0.5      # graus
_MAX_DESKEW_ANGLE = 12.0     # acima disso é provável má detecção
_MIN_CROP_AREA_RATIO = 0.30  # papel deve ocupar ≥30% da imagem pra recortar


def _decode(image_bytes: bytes) -> np.ndarray | None:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img  # None se falhar


def _encode(img: np.ndarray) -> bytes:
    """JPEG com qualidade alta — perdemos pouco vs PNG e o anthropic SDK aceita
    JPEG em base64 sem reconverter."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("cv2.imencode falhou")
    return buf.tobytes()


def _deskew_angle(gray: np.ndarray) -> float:
    """Estima ângulo dominante das linhas do papel quadriculado.

    Hough probabilístico nas bordas; retorna a mediana dos ângulos próximos a
    horizontal (|theta - 90°| < 30°). Mediana é robusta a outlier de borda do
    paciente/mão. Retorna 0.0 se sinal fraco."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    h, w = gray.shape
    # threshold proporcional ao tamanho — imagens grandes precisam de mais
    # votos pra contar como linha de verdade.
    thr = max(80, min(h, w) // 8)
    lines = cv2.HoughLines(edges, 1, np.pi / 360, threshold=thr)
    if lines is None or len(lines) < 10:
        return 0.0
    angles = []
    for rho_theta in lines[:200]:  # cap pra latência
        _, theta = rho_theta[0]
        deg = (theta * 180.0 / np.pi) - 90.0  # 0 = horizontal
        if -_MAX_DESKEW_ANGLE <= deg <= _MAX_DESKEW_ANGLE:
            angles.append(deg)
    if len(angles) < 5:
        return 0.0
    return float(np.median(angles))


def _rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotação afim com expansão da bbox (não corta cantos)."""
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(img, M, (new_w, new_h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _crop_paper(img: np.ndarray) -> np.ndarray:
    """Recorta o maior retângulo claro (papel) contra fundo escuro.

    Estratégia: binariza por Otsu invertido, dilata, pega maior contorno e
    retorna seu bbox. Se o recorte for menor que MIN_CROP_AREA_RATIO da
    imagem original, abandona (provavelmente detectou só um pedaço)."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Otsu binarizado — papel branco vira foreground (255), mesa preta vira
    # 0; sem inverter (papel já é o claro).
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # dilata pra fechar grade quadriculada que fragmenta o contorno
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    largest = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(largest)
    if (cw * ch) / (w * h) < _MIN_CROP_AREA_RATIO:
        return img
    # margem pequena pra não comer borda do traçado
    pad = 8
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + cw + pad), min(h, y + ch + pad)
    return img[y0:y1, x0:x1]


def _clahe(img: np.ndarray) -> np.ndarray:
    """CLAHE no canal L do Lab. Reforça contraste do traçado sem estourar a
    grade fininha do quadriculado. clipLimit baixo (1.5) — agressivo cria
    halo nas linhas."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_Lab2BGR)


def preprocess(image_bytes: bytes) -> bytes:
    """Aplica o pipeline e retorna JPEG resultante. Falha em qualquer estágio
    → devolve os bytes originais (pipeline da vision nunca quebra por causa
    do pré-processamento)."""
    if not PREPROCESS_VISION:
        return image_bytes
    try:
        img = _decode(image_bytes)
        if img is None:
            return image_bytes

        if PREPROCESS_DESKEW:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            angle = _deskew_angle(gray)
            if abs(angle) >= _MIN_DESKEW_ANGLE:
                img = _rotate(img, angle)
                logger.debug("preprocess: deskew {:.2f}deg", angle)

        if PREPROCESS_CROP_PAPER:
            img = _crop_paper(img)

        if PREPROCESS_CLAHE:
            img = _clahe(img)

        return _encode(img)
    except Exception as e:
        logger.warning("preprocess falhou ({}); usando imagem original", e)
        return image_bytes


def preprocess_path(path: Union[str, Path]) -> bytes:
    return preprocess(Path(path).read_bytes())
