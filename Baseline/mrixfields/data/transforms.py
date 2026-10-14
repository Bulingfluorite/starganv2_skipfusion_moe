from __future__ import annotations

"""Preprocessing transforms for MRI volumes and 2D slices."""

from typing import Optional, Tuple

import numpy as np
import torch
from scipy import ndimage


class CenterCropOrPad:
    """Center-crop or zero-pad a 3D volume or 2D slice to a target shape."""

    def __init__(self, target_shape: Tuple[int, ...]):
        self.target_shape = target_shape

    def __call__(self, data: np.ndarray) -> np.ndarray:
        result = np.zeros(self.target_shape, dtype=data.dtype)
        slices_src = []
        slices_dst = []
        for i, (s, t) in enumerate(zip(data.shape, self.target_shape)):
            if s > t:
                start = (s - t) // 2
                slices_src.append(slice(start, start + t))
                slices_dst.append(slice(None))
            else:
                start = (t - s) // 2
                slices_src.append(slice(None))
                slices_dst.append(slice(start, start + s))
        result[tuple(slices_dst)] = data[tuple(slices_src)]
        return result


class NormalizeMinMax:
    """Min-max normalize a volume to [0, 1], with optional percentile clipping.

    If `clip_percentile` is set (e.g. 99.5), values are first clipped to that
    percentile then min-max rescaled. If `clip_percentile` is None (default),
    the volume is returned as float32 unchanged.
    """

    def __init__(self, clip_percentile: Optional[float] = None):
        self.clip_percentile = clip_percentile

    def __call__(self, data: np.ndarray) -> np.ndarray:
        data = data.astype(np.float32)
        if self.clip_percentile is None:
            return data
        upper = np.percentile(data, self.clip_percentile)
        data = np.clip(data, 0, upper)
        dmin, dmax = data.min(), data.max()
        if dmax - dmin > 1e-8:
            data = (data - dmin) / (dmax - dmin)
        return data


class ToTensor:
    """Convert numpy array to PyTorch tensor, adding channel dimension."""

    def __call__(self, data: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(data.copy()).unsqueeze(0).float()


class ScaleToMinusOneOne:
    """Scale tensor from [0, 1] to [-1, 1]: x * 2 - 1.

    Standard range for GAN training (CycleGAN, CUT, StarGAN v2).
    Matches Tanh generator output range.
    """

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return data * 2.0 - 1.0


class Compose:
    """Chain multiple transforms together."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, data: np.ndarray) -> np.ndarray | torch.Tensor:
        for t in self.transforms:
            data = t(data)
        return data


class RandomRotation:
    """Random rotation by ±max_degrees * multiplier."""

    def __init__(self, max_degrees: float = 2.0):
        self.max_degrees = max_degrees
        self.multiplier = 1.0

    def __call__(self, data: np.ndarray, multiplier: float = None) -> np.ndarray:
        if multiplier is None:
            multiplier = self.multiplier
        if multiplier <= 0:
            return data
        angle = np.random.uniform(-self.max_degrees, self.max_degrees) * multiplier
        return ndimage.rotate(data, angle, order=1, mode='constant', cval=0, reshape=False)


class RandomShift:
    """Random translation by ±max_shift * multiplier pixels."""

    def __init__(self, max_shift: int = 5):
        self.max_shift = max_shift
        self.multiplier = 1.0

    def __call__(self, data: np.ndarray, multiplier: float = None) -> np.ndarray:
        if multiplier is None:
            multiplier = self.multiplier
        if multiplier <= 0:
            return data
        dx = np.random.uniform(-self.max_shift, self.max_shift) * multiplier
        dy = np.random.uniform(-self.max_shift, self.max_shift) * multiplier
        return ndimage.shift(data, [dx, dy], order=1, mode='constant', cval=0)


class RandomFlip:
    """Random horizontal flip with probability flip_prob * multiplier."""

    def __init__(self, flip_prob: float = 0.5):
        self.flip_prob = flip_prob
        self.multiplier = 1.0

    def __call__(self, data: np.ndarray, multiplier: float = None) -> np.ndarray:
        if multiplier is None:
            multiplier = self.multiplier
        if multiplier <= 0:
            return data
        if np.random.random() < self.flip_prob * multiplier:
            return np.fliplr(data)
        return data


class IntensityJitter:
    """Random brightness and contrast adjustment for intensity augmentation.

    Simulates scanner parameter variations (brightness) and tissue contrast
    differences (contrast). Applied before ToTensor on [0, 1] numpy arrays.

    Parameters:
        brightness (float) -- max relative brightness change (default 0.1)
        contrast (float)   -- max relative contrast change (default 0.1)
        p (float)          -- probability of applying the transform (default 0.5)
    """

    def __init__(self, brightness: float = 0.1, contrast: float = 0.1, p: float = 0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, data: np.ndarray) -> np.ndarray:
        if np.random.random() >= self.p:
            return data
        b = 1.0 + np.random.uniform(-self.brightness, self.brightness)
        c = 1.0 + np.random.uniform(-self.contrast, self.contrast)
        data = data * b * c
        return np.clip(data, 0.0, 1.0)
