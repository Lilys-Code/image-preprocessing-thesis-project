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
python src\cli.py --input path\to\image.jpg --output out\image_preproc.jpg
```

3. Run tests:

```bash
pytest -q
```

What is included

- `src/preprocessing` — modular preprocessing functions and `Pipeline` class
- `src/cli.py` — simple CLI to run a pipeline on an image
- `tests/test_pipeline.py` — basic unit test

