import numpy as np
import cv2

from skimage.feature import canny


def extract_canny(sar):
    """Extract a normalized Canny edge map from one SAR channel."""
    sar = sar.cpu()
    sar_edge = np.zeros_like(sar)
    image = sar.detach().numpy().astype(float)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    image = np.squeeze(image.astype(np.uint8))
    edge = canny(image, sigma=3).astype(np.float32)
    sar_edge[0, :, :] = cv2.normalize(edge, None, 0, 1, cv2.NORM_MINMAX)
    return sar_edge
