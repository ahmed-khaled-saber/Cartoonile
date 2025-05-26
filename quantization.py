import cv2
import numpy as np

def quantize_image(img: np.ndarray, k: int = 8) -> np.ndarray:
    """Reduce image colors using K-Means clustering for cartoon effect."""
    # Convert image to float32 and reshape to 2D array of pixels
    data = img.reshape((-1, 3))
    data = np.float32(data)

    # Define criteria and apply KMeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert centers to uint8 and map labels back to image
    centers = np.uint8(centers)
    quantized_data = centers[labels.flatten()]
    quantized_img = quantized_data.reshape(img.shape)

    return quantized_img
