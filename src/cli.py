import argparse
from preprocessing import load_image, save_image, build_default_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline on an image")
    parser.add_argument("--input", "-i", required=True, help="Input image path")
    parser.add_argument("--output", "-o", required=True, help="Output image path")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    args = parser.parse_args()

    img = load_image(args.input)
    pipeline = build_default_pipeline((args.width, args.height))
    out = pipeline.run(img)
    save_image(args.output, out)


if __name__ == "__main__":
    main()
"""Command-line interface to run preprocessing pipelines on folders."""
import os
import click
import cv2
from preprocessing.pipeline import Pipeline
from preprocessing.transforms import (
    resize,
    to_grayscale,
    gaussian_blur,
    normalize,
    equalize_histogram,
)
from preprocessing.utils import list_images, load_image, save_image


@click.command()
@click.option("--input", "input_dir", required=True, help="Input images directory")
@click.option("--output", "output_dir", required=True, help="Output directory")
@click.option("--resize", nargs=2, type=int, default=None, help="Resize to WIDTH HEIGHT")
@click.option("--grayscale", is_flag=True, help="Convert to grayscale")
@click.option("--blur", type=int, default=0, help="Gaussian blur kernel size (odd)")
@click.option("--equalize", is_flag=True, help="Apply histogram equalization")
def main(input_dir, output_dir, resize, grayscale, blur, equalize):
    os.makedirs(output_dir, exist_ok=True)
    files = list_images(input_dir)
    if not files:
        click.echo("No images found in input dir")
        return

    # Build pipeline based on flags
    p = Pipeline()
    if resize:
        w, h = resize
        p.add(resize, size=(w, h))
    if grayscale:
        p.add(to_grayscale)
    if blur and blur > 0:
        p.add(gaussian_blur, ksize=blur)
    if equalize:
        p.add(equalize_histogram)
    p.add(normalize)

    for path in files:
        img = load_image(path)
        if img is None:
            click.echo(f"Failed to read: {path}")
            continue
        out = p.run(img)
        rel = os.path.relpath(path, input_dir)
        dest = os.path.join(output_dir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        save_image(dest, out)

    click.echo(f"Processed {len(files)} images.")


if __name__ == "__main__":
    main()
