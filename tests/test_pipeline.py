import numpy as np
from preprocessing import Pipeline, to_grayscale, resize, normalize


def test_pipeline_basic():
    # create a synthetic color image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[25:75, 25:75] = 200

    p = Pipeline([lambda im: resize(im, (50, 50)), to_grayscale, normalize])
    out = p.run(img)

    assert out.shape == (50, 50)
    assert out.dtype == np.uint8
