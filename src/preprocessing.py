import cv2
import numpy as np

def load_image(path: str) -> np.ndarray:
    \"\"\"
    Load an image from disk.
    \"\"\"
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Unable to load image at {path}")
    return img

def crop_to_circle(img: np.ndarray) -> np.ndarray:
    \"\"\"
    Mask the image to a centered circle.
    \"\"\"
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    radius = min(center)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    return cv2.bitwise_and(img, img, mask=mask)

def preprocess_image(path: str, size: int = 300) -> np.ndarray:
    \"\"\"
    Full pipeline: load → circle crop → grayscale → resize.
    Returns a 2D array of shape (size, size).
    \"\"\"
    img = load_image(path)
    circ = crop_to_circle(img)
    gray = cv2.cvtColor(circ, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return resized

def generate_nail_positions(n_nails: int, img_size: int = 300) -> np.ndarray:
    \"\"\"
    Evenly sample n_nails points around the unit circle,
    then scale to pixel coords in [0, img_size).
    Returns an array of shape (n_nails, 2) with (x, y) coords.
    \"\"\"
    angles = np.linspace(0, 2 * np.pi, n_nails, endpoint=False)
    xs = np.cos(angles)
    ys = np.sin(angles)
    coords = np.column_stack([
        ((xs + 1) * 0.5 * (img_size - 1)),
        ((ys + 1) * 0.5 * (img_size - 1))
    ])
    return coords.astype(int)
