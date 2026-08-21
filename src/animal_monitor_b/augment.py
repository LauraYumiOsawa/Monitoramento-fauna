"""Aumentos de dados para recortes, implementados só com NumPy e OpenCV.

Evita dependências extras (Albumentations traz SciPy/Pydantic) porque o mesmo código
precisa rodar na máquina de treino sem inflar o ambiente. Os aumentos foram escolhidos
para o cenário de armadilha fotográfica: escala e enquadramento variáveis, iluminação
irregular, movimento, compressão de vídeo e captura infravermelha monocromática à noite.
"""

from __future__ import annotations

import random

import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - ambiente sem OpenCV
        raise RuntimeError("OpenCV não está instalado.") from exc
    return cv2


def resize_square(image: np.ndarray, size: int) -> np.ndarray:
    cv2 = _cv2()
    if image.shape[0] == size and image.shape[1] == size:
        return image
    interpolation = cv2.INTER_AREA if image.shape[0] > size else cv2.INTER_LINEAR
    return cv2.resize(image, (size, size), interpolation=interpolation)


def random_resized_crop(image: np.ndarray, rng: random.Random, size: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = rng.uniform(0.70, 1.0)
    ratio = rng.uniform(0.85, 1.18)
    crop_h = min(height, max(8, int(round(height * scale / ratio))))
    crop_w = min(width, max(8, int(round(width * scale * ratio))))
    top = rng.randint(0, height - crop_h)
    left = rng.randint(0, width - crop_w)
    return resize_square(image[top : top + crop_h, left : left + crop_w], size)


def random_rotate(image: np.ndarray, rng: random.Random, max_degrees: float = 12.0) -> np.ndarray:
    cv2 = _cv2()
    angle = rng.uniform(-max_degrees, max_degrees)
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    return cv2.warpAffine(
        image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )


def color_jitter(image: np.ndarray, rng: random.Random) -> np.ndarray:
    cv2 = _cv2()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-8.0, 8.0)) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(0.65, 1.35), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * rng.uniform(0.60, 1.40), 0, 255)
    jittered = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    contrast = rng.uniform(0.80, 1.25)
    mean = float(jittered.mean())
    return np.clip((jittered.astype(np.float32) - mean) * contrast + mean, 0, 255).astype(np.uint8)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Simula captura infravermelha noturna, comum em armadilhas fotográficas."""
    cv2 = _cv2()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def motion_blur(image: np.ndarray, rng: random.Random) -> np.ndarray:
    cv2 = _cv2()
    size = rng.choice((3, 5, 7))
    kernel = np.zeros((size, size), dtype=np.float32)
    if rng.random() < 0.5:
        kernel[size // 2, :] = 1.0
    else:
        kernel[:, size // 2] = 1.0
    kernel /= kernel.sum()
    return cv2.filter2D(image, -1, kernel)


def jpeg_noise(image: np.ndarray, rng: random.Random) -> np.ndarray:
    cv2 = _cv2()
    quality = rng.randint(35, 80)
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:  # pragma: no cover - falha improvável do codec
        return image
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return image if decoded is None else decoded


def gaussian_noise(image: np.ndarray, rng: random.Random) -> np.ndarray:
    sigma = rng.uniform(3.0, 12.0)
    generator = np.random.default_rng(rng.randrange(2**32))
    noise = generator.normal(0.0, sigma, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def cutout(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Oclusão parcial: vegetação na frente do animal é a regra, não a exceção."""
    height, width = image.shape[:2]
    patched = image.copy()
    for _ in range(rng.randint(1, 2)):
        box_h = rng.randint(height // 10, max(height // 10 + 1, height // 4))
        box_w = rng.randint(width // 10, max(width // 10 + 1, width // 4))
        top = rng.randint(0, height - box_h)
        left = rng.randint(0, width - box_w)
        patched[top : top + box_h, left : left + box_w] = rng.randint(0, 255)
    return patched


def augment_crop(image: np.ndarray, rng: random.Random, size: int) -> np.ndarray:
    """Pipeline de aumento aplicado a um recorte BGR uint8."""
    out = random_resized_crop(image, rng, size)
    if rng.random() < 0.50:
        out = out[:, ::-1]
    if rng.random() < 0.35:
        out = random_rotate(out, rng)
    if rng.random() < 0.70:
        out = color_jitter(out, rng)
    if rng.random() < 0.20:
        out = to_grayscale(out)
    if rng.random() < 0.20:
        out = motion_blur(out, rng)
    if rng.random() < 0.25:
        out = jpeg_noise(out, rng)
    if rng.random() < 0.20:
        out = gaussian_noise(out, rng)
    if rng.random() < 0.25:
        out = cutout(out, rng)
    return np.ascontiguousarray(out)


def to_tensor(image: np.ndarray) -> np.ndarray:
    """Converte BGR uint8 HxWx3 em CHW float32 normalizado (padrão ImageNet)."""
    rgb = np.ascontiguousarray(image[:, :, ::-1], dtype=np.float32) / 255.0
    normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1), dtype=np.float32)


def batch_to_tensor(images: list[np.ndarray]) -> np.ndarray:
    if not images:
        return np.zeros((0, 3, 1, 1), dtype=np.float32)
    return np.stack([to_tensor(image) for image in images], axis=0)
