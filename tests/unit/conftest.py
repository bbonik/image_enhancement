"""Shared fixtures: small, deterministic synthetic test images."""

import numpy as np
import pytest


@pytest.fixture
def color_uint8():
    """32x48x3 uint8 color image: smooth gradients + reproducible noise."""
    rng = np.random.RandomState(42)
    y, x = np.mgrid[0:32, 0:48]
    image = np.zeros((32, 48, 3))
    image[:, :, 0] = x / 47.0                      # horizontal ramp
    image[:, :, 1] = y / 31.0                      # vertical ramp
    image[:, :, 2] = (x + y) / (47.0 + 31.0)       # diagonal ramp
    image = image * 0.8 + rng.rand(32, 48, 3) * 0.2
    return (image * 255).astype(np.uint8)


@pytest.fixture
def gray_float():
    """32x48 float [0,1] grayscale image: gradient + reproducible noise."""
    rng = np.random.RandomState(7)
    y, x = np.mgrid[0:32, 0:48]
    image = (x + y) / (47.0 + 31.0)
    image = image * 0.7 + rng.rand(32, 48) * 0.3
    return np.clip(image, 0, 1)


@pytest.fixture
def exposure_stack(color_uint8):
    """Three color 'exposures' of the same scene (dark / mid / bright)."""
    base = color_uint8.astype(np.float64) / 255.0
    dark = np.clip(base * 0.3, 0, 1)
    mid = base
    bright = np.clip(base * 1.8, 0, 1)
    return [
        (dark * 255).astype(np.uint8),
        (mid * 255).astype(np.uint8),
        (bright * 255).astype(np.uint8),
    ]
