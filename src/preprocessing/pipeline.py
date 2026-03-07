from typing import Callable, List, Tuple
import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if img is None:
        return img
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def resize(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def gaussian_denoise(img: np.ndarray, ksize: Tuple[int, int] = (5, 5), sigma: int = 0) -> np.ndarray:
    return cv2.GaussianBlur(img, ksize, sigma)


def median_denoise(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    return cv2.medianBlur(img, ksize)


def hybrid_denoise(img: np.ndarray, window_size: int = 3) -> np.ndarray:
    """Optimized hybrid filter using vectorized NumPy operations.
    
    Uses stride tricks to avoid nested Python loops, making it significantly faster
    than the naive implementation while maintaining identical results.
    """
    if len(img.shape) == 3:
        channels = cv2.split(img)
        filtered_channels = [hybrid_denoise(ch, window_size) for ch in channels]
        return cv2.merge(filtered_channels)
    
    h, w = img.shape
    pad = window_size // 2
    padded = np.pad(img.astype(np.float32), pad, mode='constant', constant_values=0)
    
    # Use stride tricks to create a view of all windows without copying
    windows = as_strided(
        padded,
        shape=(h, w, window_size, window_size),
        strides=(padded.strides[0], padded.strides[1], padded.strides[0], padded.strides[1])
    )
    
    # Reshape to (h, w, window_size*window_size) for batch operations
    windows_flat = windows.reshape(h, w, -1)
    
    # Compute medians for all windows at once
    medians = np.median(windows_flat, axis=2, keepdims=True)
    
    # Adjust all pixels: (pixel + median) / 2
    adjusted = (windows_flat + medians) / 2.0
    
    # Compute mean of adjusted values for each window
    output = np.mean(adjusted, axis=2)
    
    return output.astype(np.uint8)


def equalize_hist(img: np.ndarray) -> np.ndarray:
    # For grayscale
    if len(img.shape) == 2:
        return cv2.equalizeHist(img)
    # For color images operate on Y channel
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def normalize(img: np.ndarray) -> np.ndarray:
    arr = img.astype('float32')
    cv2.normalize(arr, arr, 0, 255, cv2.NORM_MINMAX)
    return arr.astype('uint8')


class Pipeline:
    """A simple composable image preprocessing pipeline.

    Steps are tuples of (callable, kwargs) and are applied in order.
    Each callable must accept an image (numpy array) as first arg and
    return the transformed image.
    """

    def __init__(self, steps: List[Tuple[Callable, dict]] = None):
        # Normalize steps: accept either list of callables or list of (callable, kwargs)
        self.steps: List[Tuple[Callable, dict]] = []
        if steps:
            for s in steps:
                if callable(s):
                    self.steps.append((s, {}))
                elif isinstance(s, tuple) and callable(s[0]):
                    func = s[0]
                    kwargs = s[1] if len(s) > 1 and isinstance(s[1], dict) else {}
                    self.steps.append((func, kwargs))
                else:
                    raise TypeError("Invalid pipeline step: must be callable or (callable, dict)")

    def add(self, func: Callable, **kwargs):
        """Add a transform to the pipeline.

        Returns self to allow chaining.
        """
        self.steps.append((func, kwargs))
        return self

    def run(self, image: np.ndarray) -> np.ndarray:
        """Apply all steps to the provided image and return result."""
        img = image
        for func, kwargs in self.steps:
            img = func(img, **kwargs)
        return img

    @classmethod
    def from_transforms(cls, transforms: List[Tuple[Callable, dict]]):
        return cls(steps=list(transforms))


def build_default_pipeline(target_size: Tuple[int, int] = (256, 256)) -> Pipeline:
    p = Pipeline()
    p.add(resize, size=target_size)
    p.add(hybrid_denoise)
    # p.add(gaussian_denoise, ksize=(5, 5), sigma=0)
    # p.add(equalize_hist)
    # p.add(normalize)
    return p
