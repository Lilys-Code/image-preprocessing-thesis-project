from pathlib import Path
import cv2

def load_image(path, flags=cv2.IMREAD_COLOR):
    """Load image using OpenCV. Returns BGR image as numpy array."""
    img = cv2.imread(str(path), flags)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return img


def save_image(path, img):
    """Save image (creates parent directories if needed)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # cv2.imwrite expects a path string
    success = cv2.imwrite(str(p), img)
    if not success:
        raise IOError(f"Failed to write image to {path}")
