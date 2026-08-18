"""
src/preprocessing/processor.py

Configurable preprocessing pipeline for SEM FinFET images:
- Grayscale normalization
- Noise reduction (Bilateral, Gaussian, NLM)
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Gradient extraction (Sobel X, Sobel Y, Gradient Magnitude, Laplacian)
"""

import cv2
import numpy as np

class ImagePreprocessor:
    def __init__(self,
                 use_clahe=True,
                 clahe_clip=2.0,
                 clahe_grid=(8, 8),
                 filter_type="bilateral", # "bilateral", "gaussian", "nlm", "none"
                 bilateral_d=5,
                 bilateral_sigma_color=50,
                 bilateral_sigma_space=50,
                 gaussian_kernel=(3, 3),
                 gaussian_sigma=1.0):
        
        self.use_clahe = use_clahe
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid) if use_clahe else None
        self.filter_type = filter_type
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.gaussian_kernel = gaussian_kernel
        self.gaussian_sigma = gaussian_sigma

    def denoise(self, img_uint8):
        """Applies configured denoising filter to uint8 grayscale image."""
        if self.filter_type == "bilateral":
            return cv2.bilateralFilter(img_uint8, self.bilateral_d,
                                       self.bilateral_sigma_color,
                                       self.bilateral_sigma_space)
        elif self.filter_type == "gaussian":
            return cv2.GaussianBlur(img_uint8, self.gaussian_kernel, self.gaussian_sigma)
        elif self.filter_type == "nlm":
            return cv2.fastNlMeansDenoising(img_uint8, None, 10, 7, 21)
        else:
            return img_uint8.copy()

    def enhance_contrast(self, img_uint8):
        """Applies CLAHE if enabled."""
        if self.use_clahe and self.clahe is not None:
            return self.clahe.apply(img_uint8)
        return img_uint8.copy()

    def extract_gradients(self, img_uint8):
        """
        Extracts Sobel X, Sobel Y, Gradient Magnitude, and Laplacian features.
        Returns a dictionary of normalized float32 images in range [0, 1].
        """
        img_f = img_uint8.astype(np.float32) / 255.0
        
        sobel_x = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(sobel_x, sobel_y)
        laplacian = cv2.Laplacian(img_f, cv2.CV_32F, ksize=3)
        
        # Safe normalization to [0, 1]
        def norm(arr):
            max_v = np.max(arr)
            min_v = np.min(arr)
            if max_v > min_v:
                return (arr - min_v) / (max_v - min_v + 1e-7)
            return np.zeros_like(arr)

        return {
            "intensity": img_f,
            "sobel_x": norm(sobel_x),
            "sobel_y": norm(sobel_y),
            "gradient_magnitude": norm(grad_mag),
            "laplacian": norm(laplacian)
        }

    def process(self, img_uint8):
        """
        Full preprocessing pipeline returning both processed uint8 image and gradient maps.
        """
        denoised = self.denoise(img_uint8)
        enhanced = self.enhance_contrast(denoised)
        grads = self.extract_gradients(enhanced)
        return enhanced, grads
