"""
Core layer decomposition logic.
Separates an RGBA image into lineart, flat color, highlight, and shadow layers.

Pipeline:
  1. Gaussian blur -> flat (base color)
  2. HSV brightness comparison -> highlight / shadow
  3. Canny on flat -> lineart
"""

import cv2
import numpy as np


def decompose_layer(
    img_rgba: np.ndarray,
    blur_size: int = 10,
    canny_low: int = 50,
    canny_high: int = 150,
    brightness_threshold: int = 5,
) -> dict:
    """
    Decompose an RGBA image into 4 rendering layers.

    Args:
        img_rgba: (H, W, 4) uint8 RGBA image
        blur_size: Gaussian blur kernel half-size for flat color extraction
        canny_low: Canny edge detection low threshold
        canny_high: Canny edge detection high threshold
        brightness_threshold: Min V-channel difference to classify as HL/shadow

    Returns:
        dict with 'lineart', 'flat', 'highlight', 'shadow' keys,
        each (H, W, 4) uint8 RGBA numpy array
    """
    h, w = img_rgba.shape[:2]
    rgb = img_rgba[:, :, :3].copy()
    alpha = img_rgba[:, :, 3].copy()
    mask = alpha > 10

    # --- Step 1: Flat color via Gaussian blur ---
    kernel = blur_size * 2 + 1  # ensure odd
    blurred = cv2.GaussianBlur(rgb, (kernel, kernel), 0)

    flat = np.zeros((h, w, 4), dtype=np.uint8)
    flat[:, :, :3] = blurred
    flat[:, :, 3] = alpha

    # --- Step 2: HSV brightness comparison ---
    hsv_orig = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv_blur = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV).astype(np.float32)
    v_diff = hsv_orig[:, :, 2] - hsv_blur[:, :, 2]

    flat_f = blurred.astype(np.float32)
    orig_f = rgb.astype(np.float32)

    # --- Step 3: Highlight (Screen blend target) ---
    # Screen formula: result = 1 - (1-base)*(1-layer)/255
    # Inverse: layer = 255 - (255 - orig) * 255 / max(255 - flat, 1)
    bright_mask = (v_diff > brightness_threshold) & mask
    denom_screen = np.maximum(255.0 - flat_f, 1.0)
    screen_val = 255.0 - (255.0 - orig_f) * 255.0 / denom_screen
    screen_val = np.clip(screen_val, 0, 255).astype(np.uint8)

    highlight = np.zeros((h, w, 4), dtype=np.uint8)
    highlight[:, :, :3] = screen_val
    highlight[:, :, 3] = np.where(bright_mask, alpha, 0)

    # --- Step 4: Shadow (Multiply blend target) ---
    # Multiply formula: result = base * layer / 255
    # Inverse: layer = orig * 255 / max(flat, 1)
    shadow_mask = (v_diff < -brightness_threshold) & mask
    denom_multiply = np.maximum(flat_f, 1.0)
    multiply_val = orig_f * 255.0 / denom_multiply
    multiply_val = np.clip(multiply_val, 0, 255).astype(np.uint8)

    shadow = np.zeros((h, w, 4), dtype=np.uint8)
    shadow[:, :, :3] = multiply_val
    shadow[:, :, 3] = np.where(shadow_mask, alpha, 0)

    # --- Step 5: Lineart via Canny on flat ---
    gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)
    edges = np.where(mask, edges, 0).astype(np.uint8)

    lineart = np.zeros((h, w, 4), dtype=np.uint8)
    # Black lines with alpha = edge intensity
    lineart[:, :, 3] = edges

    return {
        "lineart": lineart,
        "flat": flat,
        "highlight": highlight,
        "shadow": shadow,
    }
