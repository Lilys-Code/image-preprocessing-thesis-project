# Image Preprocessing Pipeline

Simple, modular image preprocessing pipeline using OpenCV.

## Quick Start

1. Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Example usage (CLI):

```powershell
# single file
python src\cli.py --input data\input\image.jpg --output data\output\image_preproc.jpg

# process a directory recursively (preserves structure)
python src\cli.py --input data\input --output data\output
```

3. Run tests:

```bash
pytest -q
```

What is included

- `src/preprocessing` — modular preprocessing functions and `Pipeline` class
- `src/cli.py` — simple CLI to run a pipeline on an image
- `tests/test_pipeline.py` — basic unit test

