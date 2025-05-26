import cv2
import numpy as np

def adjust_saturation(img: np.ndarray, saturation_scale: float = 1.0) -> np.ndarray:
    """Adjust color saturation of an image using HSV color space."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= saturation_scale
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return result
