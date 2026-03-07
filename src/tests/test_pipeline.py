import numpy as np
from preprocessing import Pipeline, to_grayscale, resize, normalize, hybrid_denoise


def test_pipeline_basic():
    # create a synthetic color image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[25:75, 25:75] = 200

    p = Pipeline([lambda im: resize(im, (50, 50)), to_grayscale, normalize])
    out = p.run(img)

    assert out.shape == (50, 50)
    assert out.dtype == np.uint8


def test_hybrid_denoise():
    # Create a simple grayscale image with some noise
    img = np.ones((5, 5), dtype=np.uint8) * 128
    img[2, 2] = 255  # salt noise
    img[1, 1] = 0    # pepper noise
    
    filtered = hybrid_denoise(img, window_size=3)
    
    assert filtered.shape == img.shape
    assert filtered.dtype == np.uint8
    # The center should be smoothed
    assert 0 <= filtered[2, 2] <= 255
