import cv2
import numpy as np

def apply_bilateral_filter(img: np.ndarray, diameter=9, sigma_color=75, sigma_space=75) -> np.ndarray:
    return cv2.bilateralFilter(img, d=diameter, sigmaColor=sigma_color, sigmaSpace=sigma_space)

def detect_edges(img: np.ndarray, block_size=9, C=2) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY,
                                  block_size,
                                  C)
    return edges

def combine_edges_with_image(img: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(img, img, mask=edges)


def dilate_edges(edges: np.ndarray, thickness: int = 1) -> np.ndarray:
    """Dilate edges to control cartoon line thickness."""
    kernel = np.ones((thickness, thickness), np.uint8)
    return cv2.dilate(edges, kernel, iterations=1)
