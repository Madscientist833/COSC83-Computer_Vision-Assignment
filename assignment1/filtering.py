import numpy as np
from typing import Tuple, Union

# 10%
def convolve2d(image: np.ndarray, kernel: np.ndarray, padding_mode: str = 'constant') -> np.ndarray:
    """
    Apply 2D convolution operation on an image with a given kernel.
    
    Args:
        image: Input image (2D or 3D numpy array)
        kernel: Convolution kernel (2D numpy array)
        padding_mode: How to handle borders ('constant', 'reflect', 'replicate', etc.)
        
    Returns:
        Convolved image (same size as input)
    """
    if kernel.ndim != 2:
        raise ValueError("Kernel must be a 2D array")

    kh, kw = kernel.shape
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError("Kernel dimensions must be odd")

    pad_h = kh // 2
    pad_w = kw // 2

    pad_mode_map = {
        'replicate': 'edge',
        'nearest': 'edge',
    }
    np_pad_mode = pad_mode_map.get(padding_mode, padding_mode)

    image_f = image.astype(np.float64)
    output = np.zeros_like(image_f, dtype=np.float64)

    if image_f.ndim == 2:
        padded = np.pad(
            image_f,
            ((pad_h, pad_h), (pad_w, pad_w)),
            mode=np_pad_mode
        )
        for i in range(image_f.shape[0]):
            for j in range(image_f.shape[1]):
                region = padded[i:i + kh, j:j + kw]
                output[i, j] = np.sum(region * kernel)
    elif image_f.ndim == 3:
        for c in range(image_f.shape[2]):
            padded = np.pad(
                image_f[:, :, c],
                ((pad_h, pad_h), (pad_w, pad_w)),
                mode=np_pad_mode
            )
            for i in range(image_f.shape[0]):
                for j in range(image_f.shape[1]):
                    region = padded[i:i + kh, j:j + kw]
                    output[i, j, c] = np.sum(region * kernel)
    else:
        raise ValueError("Input image must be 2D or 3D")

    return output

#5%
def mean_filter(image: np.ndarray, kernel_size: int = 3, padding_mode: str = 'constant') -> np.ndarray:
    """
    Apply mean filtering to an image.
    
    Args:
        image: Input image
        kernel_size: Size of the kernel (e.g., 3 for 3x3, 5 for 5x5)
        padding_mode: How to handle borders
        
    Returns:
        Filtered image
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    kernel = np.ones((kernel_size, kernel_size), dtype=np.float64)
    kernel /= kernel.size
    return convolve2d(image, kernel, padding_mode=padding_mode)

#5%
def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Generate a Gaussian kernel.
    
    Args:
        size: Kernel size (must be odd)
        sigma: Standard deviation of the Gaussian
        
    Returns:
        Gaussian kernel (normalized)
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    half = size // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(x, x)

    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)
    return kernel

#5%
def gaussian_filter(image: np.ndarray, kernel_size: int = 3, sigma: float = 1.0, 
                   padding_mode: str = 'constant') -> np.ndarray:
    """
    Apply Gaussian filtering to an image.
    
    Args:
        image: Input image
        kernel_size: Size of the kernel (must be odd)
        sigma: Standard deviation of the Gaussian
        padding_mode: How to handle borders
        
    Returns:
        Filtered image
    """
    kernel = gaussian_kernel(kernel_size, sigma)
    return convolve2d(image, kernel, padding_mode=padding_mode)

#5%
def laplacian_filter(image: np.ndarray, kernel_type: str = 'standard', 
                    padding_mode: str = 'constant') -> np.ndarray:
    """
    Apply Laplacian filtering for edge detection.
    
    Args:
        image: Input image
        kernel_type: Type of Laplacian kernel ('standard', 'diagonal')
        padding_mode: How to handle borders
        
    Returns:
        Filtered image
    """
    kernels = {
        'standard': np.array([[0, 1, 0],
                              [1, -4, 1],
                              [0, 1, 0]], dtype=np.float64),
        'diagonal': np.array([[1, 1, 1],
                              [1, -8, 1],
                              [1, 1, 1]], dtype=np.float64),
    }

    if kernel_type not in kernels:
        raise ValueError("kernel_type must be 'standard' or 'diagonal'")

    if image.ndim == 3:
        image = np.mean(image, axis=2)

    return convolve2d(image, kernels[kernel_type], padding_mode=padding_mode)

#10%
def sobel_filter(image: np.ndarray, direction: str = 'both', kernel_size: int = 3, 
                padding_mode: str = 'constant') -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Apply Sobel filtering for edge detection.
    
    Args:
        image: Input image
        direction: Direction of the filter ('x', 'y', or 'both')
        kernel_size: Size of the kernel (3, 5, etc.)
        padding_mode: How to handle borders
        
    Returns:
        If direction is 'both', returns (gradient_magnitude, gradient_direction)
        Otherwise, returns the filtered image
    """
    if image.ndim == 3:
        image = np.mean(image, axis=2)

    if kernel_size == 3:
        kx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float64)
        ky = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]], dtype=np.float64)
    elif kernel_size == 5:
        kx = np.array([
            [-1, -2, 0, 2, 1],
            [-4, -8, 0, 8, 4],
            [-6, -12, 0, 12, 6],
            [-4, -8, 0, 8, 4],
            [-1, -2, 0, 2, 1],
        ], dtype=np.float64)
        ky = np.array([
            [-1, -4, -6, -4, -1],
            [-2, -8, -12, -8, -2],
            [0, 0, 0, 0, 0],
            [2, 8, 12, 8, 2],
            [1, 4, 6, 4, 1],
        ], dtype=np.float64)
    else:
        raise ValueError("kernel_size must be 3 or 5")

    if direction == 'x':
        return convolve2d(image, kx, padding_mode=padding_mode)
    if direction == 'y':
        return convolve2d(image, ky, padding_mode=padding_mode)
    if direction != 'both':
        raise ValueError("direction must be 'x', 'y', or 'both'")

    gx = convolve2d(image, kx, padding_mode=padding_mode)
    gy = convolve2d(image, ky, padding_mode=padding_mode)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    angle = np.arctan2(gy, gx)
    return magnitude, angle

# These helper functions are provided for you

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image values to range [0, 255] and convert to uint8.
    """
    min_val = np.min(image)
    max_val = np.max(image)
    
    # Check to avoid division by zero
    if max_val == min_val:
        return np.zeros_like(image, dtype=np.uint8)
    
    # Normalize to [0, 255]
    normalized = 255 * (image - min_val) / (max_val - min_val)
    return normalized.astype(np.uint8)


def add_noise(image: np.ndarray, noise_type: str = 'gaussian', var: float = 0.01) -> np.ndarray:
    """
    Add noise to an image.
    
    Args:
        image: Input image
        noise_type: Type of noise ('gaussian' or 'salt_pepper')
        var: Variance (for Gaussian) or density (for salt and pepper)
        
    Returns:
        Noisy image
    """
    image_copy = image.copy().astype(np.float32)
    
    if noise_type == 'gaussian':
        # Add Gaussian noise
        noise = np.random.normal(0, var**0.5, image.shape)
        noisy = image_copy + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    elif noise_type == 'salt_pepper':
        # Add salt and pepper noise
        salt_mask = np.random.random(image.shape) < var/2
        pepper_mask = np.random.random(image.shape) < var/2
        
        noisy = image_copy.copy()
        noisy[salt_mask] = 255
        noisy[pepper_mask] = 0
        return noisy.astype(np.uint8)
    
    else:
        raise ValueError("Unknown noise type. Use 'gaussian' or 'salt_pepper'")