"""A fixed, deterministic corruption suite for measuring robustness.

Why this exists
---------------
Iteration 3 is described in ``README.md`` as having "drastically higher
real-world generalization" than iteration 2. That claim was never measured.
Augmentation was applied to *training only*, and the two models were compared on
the same clean test split -- on which iteration 3's validation accuracy was
actually *lower* (88.77% vs 89.16%). So the project's headline robustness claim
was, at best, unevidenced.

This module supplies the missing measurement: a set of image corruptions applied
to the **test** split at inference time, so the same two checkpoints can be
compared under degradation without retraining anything.

Design decisions
----------------
- **Deterministic.** Each corruption is a pure function of (image, severity).
  There is no sampling, so re-running produces identical inputs and the
  comparison between models is exact rather than approximate.
- **Applied before resize/normalize**, on the uint8 RGB image, so severity means
  the same thing regardless of a model's input resolution.
- **Severities 1-3**, chosen so that severity 1 is a mild real-world nuisance and
  severity 3 is bad-but-not-absurd. Reporting a curve rather than one point
  shows *where* a model degrades, which is the interesting result.
- **Disjoint from training augmentation where it matters.** ``fog`` overlaps
  conceptually with the ``RandomFog`` that iteration 3 trained on, so it is
  expected to favour iteration 3. That is itself worth reporting: it separates
  "robust to what it saw" from "robust in general". The corruptions iteration 3
  never trained on (motion blur, JPEG artifacts, Gaussian noise) are the honest
  generalization test, and results are grouped accordingly.
"""

from __future__ import annotations

import io
import logging
from typing import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)

SEVERITIES = (1, 2, 3)

#: Corruptions iteration 3 trained on an analogue of. Expected to favour it.
SEEN_IN_TRAINING = {"fog", "brightness_up", "brightness_down", "contrast_down"}

#: Corruptions absent from every training pipeline. The real generalization test.
UNSEEN_IN_TRAINING = {"motion_blur", "gaussian_noise", "jpeg", "defocus_blur"}


def _as_pil(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    return image


def fog(image: Image.Image, severity: int) -> Image.Image:
    """
    Blend the image toward a bright grey veil, mimicking haze or thin smoke.

    Deliberately close to Albumentations' ``RandomFog`` in effect, because the
    point is to test whether training on synthetic fog transfers.
    """
    coefficients = {1: 0.15, 2: 0.30, 3: 0.45}
    alpha = coefficients[severity]
    array = np.asarray(_as_pil(image).convert("RGB"), dtype=np.float32)
    veil = np.full_like(array, 220.0)
    blended = (1.0 - alpha) * array + alpha * veil
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def brightness_up(image: Image.Image, severity: int) -> Image.Image:
    """Overexposure, as from a camera pointed near the sun."""
    factors = {1: 1.3, 2: 1.6, 3: 2.0}
    return ImageEnhance.Brightness(_as_pil(image).convert("RGB")).enhance(factors[severity])


def brightness_down(image: Image.Image, severity: int) -> Image.Image:
    """Underexposure, as at dusk or night."""
    factors = {1: 0.7, 2: 0.5, 3: 0.35}
    return ImageEnhance.Brightness(_as_pil(image).convert("RGB")).enhance(factors[severity])


def contrast_down(image: Image.Image, severity: int) -> Image.Image:
    """Washed-out low-contrast capture, as from a cheap or dirty sensor."""
    factors = {1: 0.7, 2: 0.5, 3: 0.3}
    return ImageEnhance.Contrast(_as_pil(image).convert("RGB")).enhance(factors[severity])


def motion_blur(image: Image.Image, severity: int) -> Image.Image:
    """
    Horizontal streak, as from a panning or vehicle-mounted camera.

    Implemented as a box filter along one axis so the result is deterministic
    and does not depend on an RNG-seeded kernel.
    """
    kernel_sizes = {1: 5, 2: 9, 3: 15}
    size = kernel_sizes[severity]
    array = np.asarray(_as_pil(image).convert("RGB"), dtype=np.float32)

    padded = np.pad(array, ((0, 0), (size // 2, size // 2), (0, 0)), mode="edge")
    accumulator = np.zeros_like(array)
    for offset in range(size):
        accumulator += padded[:, offset : offset + array.shape[1], :]
    accumulator /= size

    return Image.fromarray(np.clip(accumulator, 0, 255).astype(np.uint8))


def defocus_blur(image: Image.Image, severity: int) -> Image.Image:
    """Out-of-focus lens, as from a fixed camera with a drifting focal plane."""
    radii = {1: 1.5, 2: 3.0, 3: 5.0}
    return _as_pil(image).convert("RGB").filter(ImageFilter.GaussianBlur(radii[severity]))


def gaussian_noise(image: Image.Image, severity: int) -> Image.Image:
    """
    Sensor noise, as from a low-light or low-quality camera.

    Uses a fixed-seed generator so the noise field is identical for every model
    and every run; the comparison must not depend on which model was evaluated
    first.
    """
    scales = {1: 8.0, 2: 18.0, 3: 32.0}
    array = np.asarray(_as_pil(image).convert("RGB"), dtype=np.float32)
    generator = np.random.default_rng(seed=1234)
    noise = generator.normal(0.0, scales[severity], array.shape)
    return Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8))


def jpeg(image: Image.Image, severity: int) -> Image.Image:
    """Compression artifacts, as from bandwidth-limited streaming off a device."""
    qualities = {1: 30, 2: 15, 3: 7}
    buffer = io.BytesIO()
    _as_pil(image).convert("RGB").save(buffer, format="JPEG", quality=qualities[severity])
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB").copy()


#: Registry of every corruption, keyed by the name used in results files.
CORRUPTIONS: dict[str, Callable[[Image.Image, int], Image.Image]] = {
    "fog": fog,
    "brightness_up": brightness_up,
    "brightness_down": brightness_down,
    "contrast_down": contrast_down,
    "motion_blur": motion_blur,
    "defocus_blur": defocus_blur,
    "gaussian_noise": gaussian_noise,
    "jpeg": jpeg,
}


def apply_corruption(image: Image.Image, name: str, severity: int) -> Image.Image:
    """Apply one named corruption at one severity."""
    if name == "none":
        return _as_pil(image).convert("RGB")
    if name not in CORRUPTIONS:
        raise ValueError(f"Unknown corruption {name!r}. Choose from {sorted(CORRUPTIONS)}.")
    if severity not in SEVERITIES:
        raise ValueError(f"Severity must be one of {SEVERITIES}, got {severity}.")
    return CORRUPTIONS[name](image, severity)


def corruption_group(name: str) -> str:
    """Classify a corruption by whether iteration 3 trained on an analogue."""
    if name in SEEN_IN_TRAINING:
        return "seen_in_training"
    if name in UNSEEN_IN_TRAINING:
        return "unseen_in_training"
    return "clean"
