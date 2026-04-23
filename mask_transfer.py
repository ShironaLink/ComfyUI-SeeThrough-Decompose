"""
High-resolution mask transfer module.
Applies See-through's low-res masks to the original high-res image,
preserving the original pixel quality.

Pipeline:
  See-through (1280px) -> masks (alpha channels)
  Original image (3000px+)
  Mask upscale + smooth -> apply to original -> high-res parts
"""

import cv2
import numpy as np
from PIL import Image


def extract_mask(part_rgba: np.ndarray) -> np.ndarray:
    """Extract alpha channel as mask from RGBA image."""
    return part_rgba[:, :, 3].copy()


def upscale_mask(
    mask: np.ndarray,
    target_h: int,
    target_w: int,
    edge_blur: int = 3,
) -> np.ndarray:
    """
    Upscale a binary/soft mask to target resolution with edge smoothing.

    Args:
        mask: (H, W) uint8 mask from See-through
        target_h: target height (original image)
        target_w: target width (original image)
        edge_blur: Gaussian blur kernel half-size for edge smoothing (0=no blur)

    Returns:
        (target_h, target_w) uint8 upscaled mask
    """
    # Use INTER_CUBIC for smoother edges than NEAREST
    upscaled = cv2.resize(
        mask, (target_w, target_h), interpolation=cv2.INTER_CUBIC
    )
    upscaled = np.clip(upscaled, 0, 255).astype(np.uint8)

    # Smooth edges to reduce staircase artifacts
    if edge_blur > 0:
        kernel = edge_blur * 2 + 1

        # Only blur the edge region, keep interior solid
        # Detect edge pixels (where mask transitions from 0 to 255)
        dilated = cv2.dilate(upscaled, None, iterations=edge_blur)
        eroded = cv2.erode(upscaled, None, iterations=edge_blur)
        edge_region = (dilated > 0) & (eroded < 255)

        blurred = cv2.GaussianBlur(upscaled, (kernel, kernel), 0)

        # Replace only edge region with blurred version
        result = upscaled.copy()
        result[edge_region] = blurred[edge_region]
        return result

    return upscaled


def apply_mask_to_original(
    original_rgba: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Apply mask to original image, producing a high-res RGBA part.

    Args:
        original_rgba: (H, W, 4) uint8, original high-res image
        mask: (H, W) uint8, upscaled mask (same size as original)

    Returns:
        (H, W, 4) uint8 RGBA with original RGB and mask as alpha
    """
    result = original_rgba.copy()
    result[:, :, 3] = np.minimum(original_rgba[:, :, 3], mask)
    return result


def transfer_masks_to_original(
    original_path: str,
    parts_dir: str,
    edge_blur: int = 3,
) -> list:
    """
    Transfer all See-through part masks to the original high-res image.

    Args:
        original_path: path to original high-res PNG
        parts_dir: directory with See-through output PNGs (low-res parts)
        edge_blur: edge smoothing amount (0=off, 1-5 recommended)

    Returns:
        list of dicts with 'tag', 'img' (RGBA numpy), ready for decomposition
    """
    import os

    # Load original at full resolution
    original = Image.open(original_path).convert("RGBA")
    orig_rgba = np.array(original)
    orig_h, orig_w = orig_rgba.shape[:2]

    # Process each part
    parts = []
    png_files = sorted(
        [f for f in os.listdir(parts_dir) if f.lower().endswith(".png")]
    )

    for fname in png_files:
        path = os.path.join(parts_dir, fname)
        part_img = Image.open(path).convert("RGBA")
        part_rgba = np.array(part_img)

        # Skip fully transparent
        if part_rgba[:, :, 3].max() < 10:
            continue

        # Extract and upscale mask
        mask = extract_mask(part_rgba)
        mask_hires = upscale_mask(mask, orig_h, orig_w, edge_blur=edge_blur)

        # Apply to original
        hires_part = apply_mask_to_original(orig_rgba, mask_hires)

        tag = os.path.splitext(fname)[0]
        parts.append({"tag": tag, "img": hires_part})

    return parts
