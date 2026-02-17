# Image Preprocessing Pipeline

Simple, modular image preprocessing pipeline using OpenCV.

Setup

1. Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Example usage (CLI):

```powershell
python src\cli.py --input path\to\image.jpg --output out\image_preproc.jpg
```

What is included

- `src/preprocessing` — modular preprocessing functions and `Pipeline` class
- `src/cli.py` — simple CLI to run a pipeline on an image
- `tests/test_pipeline.py` — basic unit test
# Image Preprocessing Pipeline

A small Python package for building image preprocessing pipelines using OpenCV.

Quick start

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Use the CLI to process a directory of images:

```bash
python -m src.cli --input data/input --output data/output --resize 512 512
```

3. Run tests:

```bash
pytest -q
```

Package layout

- `src/preprocessing` — core pipeline and transforms
- `src/cli.py` — small CLI to run preprocessing over a folder

See docs in the `src/preprocessing` package for API examples.
