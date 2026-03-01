import argparse
import os
from preprocessing import load_image, save_image, build_default_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run preprocessing pipeline on an image or directory of images"
    )
    parser.add_argument("--input", "-i", required=True, help="Input file or directory")
    parser.add_argument("--output", "-o", required=True, help="Output file or directory")
    parser.add_argument("--width", type=int, default=256, help="Output width")
    parser.add_argument("--height", type=int, default=256, help="Output height")
    args = parser.parse_args()

    pipeline = build_default_pipeline((args.width, args.height))

    inp = args.input
    outp = args.output

    if os.path.isdir(inp):
        # process recursively
        for root, dirs, files in os.walk(inp):
            rel_root = os.path.relpath(root, inp)
            for fname in files:
                in_file = os.path.join(root, fname)
                try:
                    img = load_image(in_file)
                except FileNotFoundError:
                    # skip non-image or unreadable
                    continue
                result = pipeline.run(img)
                # determine destination path
                dest_dir = os.path.join(outp, rel_root) if rel_root != '.' else outp
                os.makedirs(dest_dir, exist_ok=True)
                dest_path = os.path.join(dest_dir, fname)
                save_image(dest_path, result)
    else:
        # single file mode
        img = load_image(inp)
        result = pipeline.run(img)
        save_image(outp, result)


if __name__ == "__main__":
    main()
