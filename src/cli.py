import argparse
from preprocessing import load_image, save_image, build_default_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline on an image")
    parser.add_argument("--input", "-i", required=True, help="Input image path")
    parser.add_argument("--output", "-o", required=True, help="Output image path")
    parser.add_argument("--width", type=int, default=256, help="Output width")
    parser.add_argument("--height", type=int, default=256, help="Output height")
    args = parser.parse_args()

    img = load_image(args.input)
    pipeline = build_default_pipeline((args.width, args.height))
    out = pipeline.run(img)
    save_image(args.output, out)


if __name__ == "__main__":
    main()
