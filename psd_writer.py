"""
PSD file writer for decomposed layers.
Uses pytoshop for blend mode support (Screen, Multiply).
Falls back to psd-tools if pytoshop is unavailable.
"""

import numpy as np
from PIL import Image

# Blend mode constants for layer naming fallback
BLEND_LABELS = {
    "flat": "Normal",
    "shadow": "Multiply",
    "highlight": "Screen",
    "lineart": "Normal",
}

# Sublayer order (bottom to top in PSD)
SUBLAYER_ORDER = ["flat", "shadow", "highlight", "lineart"]


def _save_with_pytoshop(parts_list, output_path, canvas_h, canvas_w):
    """Save PSD using pytoshop (supports blend modes)."""
    import pytoshop
    from pytoshop import layers as psd_layers
    from pytoshop.enums import BlendMode, ColorMode

    blend_map = {
        "flat": BlendMode.normal,
        "shadow": BlendMode.multiply,
        "highlight": BlendMode.screen,
        "lineart": BlendMode.normal,
    }

    psd = pytoshop.PsdFile(num_channels=4, height=canvas_h, width=canvas_w)
    psd.header.color_mode = ColorMode.rgb

    # Sort parts by depth (back to front)
    sorted_parts = sorted(parts_list, key=lambda x: x.get("depth_median", 0.5))

    for part in sorted_parts:
        tag = part["tag"]
        layers_data = part["layers"]

        for layer_type in SUBLAYER_ORDER:
            img = layers_data.get(layer_type)
            if img is None:
                continue

            # Pad to canvas size
            img = _pad_to_canvas(img, canvas_h, canvas_w)
            r, g, b, a = img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]

            layer_name = f"{tag}_{layer_type}"
            layer = psd_layers.ChannelImageData.from_image(
                layer_name=layer_name,
                transparency=a,
                red=r,
                green=g,
                blue=b,
            )

            # Set blend mode
            try:
                layer.blend_mode = blend_map.get(layer_type, BlendMode.normal)
            except (AttributeError, TypeError):
                pass

            psd.layer_and_mask_info.layer_info.layer_records.append(layer)

    with open(output_path, "wb") as f:
        psd.write(f)


def _save_with_psdtools(parts_list, output_path, canvas_h, canvas_w):
    """Save PSD using psd-tools (fallback, no blend mode support)."""
    from psd_tools import PSDImage

    psd = PSDImage.new(mode="RGBA", size=(canvas_w, canvas_h), depth=8)

    sorted_parts = sorted(parts_list, key=lambda x: x.get("depth_median", 0.5))

    for part in sorted_parts:
        tag = part["tag"]
        layers_data = part["layers"]

        for layer_type in SUBLAYER_ORDER:
            img = layers_data.get(layer_type)
            if img is None:
                continue

            img = _pad_to_canvas(img, canvas_h, canvas_w)
            pil_img = Image.fromarray(img, "RGBA")

            # Include blend mode hint in layer name
            blend_hint = BLEND_LABELS.get(layer_type, "Normal")
            layer_name = f"{tag}_{layer_type} [{blend_hint}]"

            psd.create_pixel_layer(
                pil_img, name=layer_name, top=0, left=0, opacity=255
            )

    psd.save(output_path)


def _pad_to_canvas(img, canvas_h, canvas_w):
    """Pad image to canvas size if needed."""
    h, w = img.shape[:2]
    if h == canvas_h and w == canvas_w:
        return img
    padded = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
    ph, pw = min(h, canvas_h), min(w, canvas_w)
    padded[:ph, :pw] = img[:ph, :pw]
    return padded


def save_decomposed_psd(parts_list, output_path, canvas_h, canvas_w):
    """
    Save decomposed layers as PSD file.

    Args:
        parts_list: list of dicts, each with:
            'tag': str (part name, e.g. 'face', 'hair_front')
            'layers': dict with 'lineart', 'flat', 'highlight', 'shadow'
                      (each H,W,4 uint8 RGBA numpy array)
            'depth_median': float (0.0=back, 1.0=front) for layer ordering
        output_path: str, path to save PSD file
        canvas_h: int, PSD canvas height
        canvas_w: int, PSD canvas width
    """
    try:
        _save_with_pytoshop(parts_list, output_path, canvas_h, canvas_w)
        print(f"[SeeThrough-Decompose] PSD saved with pytoshop: {output_path}")
    except ImportError:
        _save_with_psdtools(parts_list, output_path, canvas_h, canvas_w)
        print(
            f"[SeeThrough-Decompose] PSD saved with psd-tools (blend modes not set): "
            f"{output_path}"
        )
