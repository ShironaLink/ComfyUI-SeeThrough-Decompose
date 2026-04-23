"""
Upscaling module for part images.
Supports Lanczos (built-in) and Real-ESRGAN (if available).
Handles RGBA properly by processing RGB and alpha separately.
"""

import cv2
import numpy as np
from PIL import Image


def upscale_rgba_lanczos(img_rgba: np.ndarray, scale: int) -> np.ndarray:
    """Upscale RGBA image using Lanczos interpolation."""
    h, w = img_rgba.shape[:2]
    new_h, new_w = h * scale, w * scale

    rgb = img_rgba[:, :, :3]
    alpha = img_rgba[:, :, 3]

    rgb_up = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    alpha_up = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    result = np.zeros((new_h, new_w, 4), dtype=np.uint8)
    result[:, :, :3] = rgb_up
    result[:, :, 3] = alpha_up
    return result


def upscale_rgba_esrgan(img_rgba: np.ndarray, scale: int, model_path: str) -> np.ndarray:
    """
    Upscale RGBA image using Real-ESRGAN.
    Alpha channel is upscaled separately with Lanczos.
    """
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ImportError:
        print("[SeeThrough-Decompose] Real-ESRGAN not installed, falling back to Lanczos")
        return upscale_rgba_lanczos(img_rgba, scale)

    h, w = img_rgba.shape[:2]
    rgb = img_rgba[:, :, :3]
    alpha = img_rgba[:, :, 3]

    # Setup Real-ESRGAN model
    if scale == 2:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    else:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        half=True,
        tile=512,
        tile_pad=10,
    )

    # Upscale RGB with Real-ESRGAN (expects BGR)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr_up, _ = upsampler.enhance(bgr, outscale=scale)
    rgb_up = cv2.cvtColor(bgr_up, cv2.COLOR_BGR2RGB)

    # Upscale alpha with Lanczos
    new_h, new_w = rgb_up.shape[:2]
    alpha_up = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    result = np.zeros((new_h, new_w, 4), dtype=np.uint8)
    result[:, :, :3] = rgb_up
    result[:, :, 3] = alpha_up
    return result


def upscale_rgba(
    img_rgba: np.ndarray,
    scale: int = 2,
    method: str = "lanczos",
    model_path: str = "",
) -> np.ndarray:
    """
    Upscale RGBA image.

    Args:
        img_rgba: (H, W, 4) uint8
        scale: 2 or 4
        method: "lanczos" or "esrgan"
        model_path: path to Real-ESRGAN model (only for esrgan method)

    Returns:
        Upscaled (H*scale, W*scale, 4) uint8 RGBA
    """
    if method == "esrgan" and model_path:
        return upscale_rgba_esrgan(img_rgba, scale, model_path)
    return upscale_rgba_lanczos(img_rgba, scale)
