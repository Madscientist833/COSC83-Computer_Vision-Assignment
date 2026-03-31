import numpy as np
from filtering import gaussian_filter, sobel_filter

#10%bonus
def canny_edge_detector(image: np.ndarray, low_threshold: float = 0.05, high_threshold: float = 0.15, 
                      sigma: float = 1.0) -> np.ndarray:
    """
    Implement the Canny edge detection algorithm from scratch.
    
    Args:
        image: Input grayscale image
        low_threshold: Low threshold for hysteresis (as a fraction of the maximum gradient magnitude)
        high_threshold: High threshold for hysteresis (as a fraction of the maximum gradient magnitude)
        sigma: Standard deviation for Gaussian filter
        
    Returns:
        Binary edge map
    """
    if image.ndim == 3:
        image = np.mean(image, axis=2)

    image = image.astype(np.float64)

    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if not (0 <= low_threshold <= 1 and 0 <= high_threshold <= 1):
        raise ValueError("Thresholds must be in [0, 1]")
    if low_threshold > high_threshold:
        raise ValueError("low_threshold must be <= high_threshold")

    kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    smoothed = gaussian_filter(image, kernel_size=kernel_size, sigma=sigma, padding_mode='reflect')

    grad_mag, grad_dir = sobel_filter(smoothed, direction='both', kernel_size=3, padding_mode='reflect')

    # Non-maximum suppression
    angle = np.rad2deg(grad_dir)
    angle[angle < 0] += 180
    nms = np.zeros_like(grad_mag)
    h, w = grad_mag.shape

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            q = 0.0
            r = 0.0

            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = grad_mag[i, j + 1]
                r = grad_mag[i, j - 1]
            elif 22.5 <= angle[i, j] < 67.5:
                q = grad_mag[i + 1, j - 1]
                r = grad_mag[i - 1, j + 1]
            elif 67.5 <= angle[i, j] < 112.5:
                q = grad_mag[i + 1, j]
                r = grad_mag[i - 1, j]
            elif 112.5 <= angle[i, j] < 157.5:
                q = grad_mag[i - 1, j - 1]
                r = grad_mag[i + 1, j + 1]

            if grad_mag[i, j] >= q and grad_mag[i, j] >= r:
                nms[i, j] = grad_mag[i, j]

    max_val = np.max(nms)
    if max_val == 0:
        return np.zeros_like(image, dtype=np.uint8)

    high = high_threshold * max_val
    low = low_threshold * max_val

    strong = 255
    weak = 75
    edges = np.zeros_like(nms, dtype=np.uint8)
    strong_coords = nms >= high
    weak_coords = (nms >= low) & (nms < high)
    edges[strong_coords] = strong
    edges[weak_coords] = weak

    # Hysteresis: keep weak edges connected to strong edges
    stack = list(zip(*np.nonzero(strong_coords)))
    while stack:
        y, x = stack.pop()
        y0 = max(y - 1, 0)
        y1 = min(y + 2, h)
        x0 = max(x - 1, 0)
        x1 = min(x + 2, w)

        for ny in range(y0, y1):
            for nx in range(x0, x1):
                if edges[ny, nx] == weak:
                    edges[ny, nx] = strong
                    stack.append((ny, nx))

    edges[edges != strong] = 0
    return edges.astype(np.uint8)