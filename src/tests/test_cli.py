import os
import sys
import numpy as np

import cv2

from preprocessing import load_image

import cli

def test_cli_directory(tmp_path, monkeypatch):
    # create an input structure with two images in nested folders
    input_dir = tmp_path / "in"
    subdir = input_dir / "sub"
    subdir.mkdir(parents=True)

    img = np.full((20, 20, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(input_dir / "a.jpg"), img)
    cv2.imwrite(str(subdir / "b.png"), img)

    output_dir = tmp_path / "out"

    # run CLI by manipulating sys.argv
    orig_argv = sys.argv.copy()
    sys.argv = ["cli.py", "--input", str(input_dir), "--output", str(output_dir)]
    try:
        cli.main()
    finally:
        sys.argv = orig_argv

    # verify outputs preserved structure
    out_a = output_dir / "a.jpg"
    out_b = output_dir / "sub" / "b.png"
    assert out_a.exists(), "output file a.jpg not created"
    assert out_b.exists(), "output file in sub folder not created"

    # load and check shape matches resized default (256x256 grayscale?)
    data_a = load_image(out_a)
    assert data_a.shape[0] == 256 or data_a.shape[1] == 256
