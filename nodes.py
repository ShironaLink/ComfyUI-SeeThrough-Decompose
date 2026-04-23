"""
ComfyUI custom nodes for SeeThrough-Decompose pipeline.

Nodes:
  STR_DecomposeLayer    - Single IMAGE -> 4 layer IMAGEs
  STR_DecomposeFolder   - Folder of parts -> STR_DECOMPOSED_DATA
  STR_UpscaleFolder     - Upscale all PNGs in folder before decomposition
  STR_SaveDecomposedPSD - STR_DECOMPOSED_DATA -> PSD file
"""

import os
import json
import glob
import torch
import numpy as np
from PIL import Image

from .decomposer import decompose_layer
from .psd_writer import save_decomposed_psd
from .upscaler import upscale_rgba


# ---------------------------------------------------------------------------
# Custom data container
# ---------------------------------------------------------------------------
class DecomposedPartsData:
    """Container for decomposed parts, passed between nodes."""

    def __init__(self, parts_list, canvas_h, canvas_w):
        self.parts_list = parts_list  # list of dicts
        self.canvas_h = canvas_h
        self.canvas_w = canvas_w


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _tensor_to_rgba(tensor):
    """Convert ComfyUI IMAGE tensor (B,H,W,C) float[0,1] to RGBA uint8 numpy."""
    img = tensor[0].cpu().numpy()  # first in batch
    img = (img * 255).clip(0, 255).astype(np.uint8)
    if img.shape[2] == 3:
        alpha = np.full((*img.shape[:2], 1), 255, dtype=np.uint8)
        img = np.concatenate([img, alpha], axis=2)
    return img


def _rgba_to_tensor(img_rgba):
    """Convert RGBA uint8 numpy (H,W,4) to ComfyUI IMAGE tensor (1,H,W,4)."""
    t = torch.from_numpy(img_rgba.astype(np.float32) / 255.0)
    return t.unsqueeze(0)


# ---------------------------------------------------------------------------
# Node 1: Single image decomposition
# ---------------------------------------------------------------------------
class STR_DecomposeLayer:
    """Decompose a single image into lineart, flat, highlight, shadow layers."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "blur_size": (
                    "INT",
                    {"default": 10, "min": 1, "max": 50, "step": 1},
                ),
                "canny_low": (
                    "INT",
                    {"default": 50, "min": 0, "max": 255, "step": 1},
                ),
                "canny_high": (
                    "INT",
                    {"default": 150, "min": 0, "max": 255, "step": 1},
                ),
                "brightness_threshold": (
                    "INT",
                    {"default": 5, "min": 1, "max": 50, "step": 1},
                ),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("lineart", "flat", "highlight", "shadow")
    FUNCTION = "decompose"
    CATEGORY = "SeeThrough-Re"

    def decompose(
        self, image, blur_size, canny_low, canny_high, brightness_threshold, mask=None
    ):
        img = image[0].cpu().numpy()
        img = (img * 255).clip(0, 255).astype(np.uint8)

        # Build RGBA
        if img.shape[2] == 3:
            if mask is not None:
                m = mask[0].cpu().numpy()
                alpha = (m * 255).clip(0, 255).astype(np.uint8)
            else:
                alpha = np.full(img.shape[:2], 255, dtype=np.uint8)
            img_rgba = np.concatenate([img, alpha[:, :, None]], axis=2)
        else:
            img_rgba = img

        result = decompose_layer(
            img_rgba,
            blur_size=blur_size,
            canny_low=canny_low,
            canny_high=canny_high,
            brightness_threshold=brightness_threshold,
        )

        return (
            _rgba_to_tensor(result["lineart"]),
            _rgba_to_tensor(result["flat"]),
            _rgba_to_tensor(result["highlight"]),
            _rgba_to_tensor(result["shadow"]),
        )


# ---------------------------------------------------------------------------
# Node 2: Batch decompose from folder
# ---------------------------------------------------------------------------
class STR_DecomposeFolder:
    """
    Read See-through output folder, optionally upscale, then decompose
    every part PNG into 4 layers.
    Outputs STR_DECOMPOSED_DATA for SavePSD node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_dir": ("STRING", {"default": ""}),
                "blur_size": (
                    "INT",
                    {"default": 10, "min": 1, "max": 50, "step": 1},
                ),
                "canny_low": (
                    "INT",
                    {"default": 50, "min": 0, "max": 255, "step": 1},
                ),
                "canny_high": (
                    "INT",
                    {"default": 150, "min": 0, "max": 255, "step": 1},
                ),
                "brightness_threshold": (
                    "INT",
                    {"default": 5, "min": 1, "max": 50, "step": 1},
                ),
                "upscale": (
                    ["none", "2x_lanczos", "4x_lanczos", "2x_esrgan", "4x_esrgan"],
                    {"default": "none"},
                ),
            },
            "optional": {
                "esrgan_model_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STR_DECOMPOSED_DATA", "IMAGE")
    RETURN_NAMES = ("decomposed_data", "preview")
    FUNCTION = "decompose_folder"
    CATEGORY = "SeeThrough-Re"

    def decompose_folder(
        self,
        input_dir,
        blur_size,
        canny_low,
        canny_high,
        brightness_threshold,
        upscale="none",
        esrgan_model_path="",
    ):
        if not os.path.isdir(input_dir):
            raise ValueError(f"Directory not found: {input_dir}")

        # Parse upscale config
        up_scale = 1
        up_method = "lanczos"
        if upscale != "none":
            parts = upscale.split("_")
            up_scale = int(parts[0].replace("x", ""))
            up_method = parts[1] if len(parts) > 1 else "lanczos"

        # Look for depth/order info JSON
        depth_info = {}
        for f in os.listdir(input_dir):
            if f.endswith(".json"):
                json_path = os.path.join(input_dir, f)
                try:
                    with open(json_path, "r", encoding="utf-8") as jf:
                        data = json.load(jf)
                    if "parts" in data:
                        for tag, info in data["parts"].items():
                            if "depth_median" in info:
                                depth_info[tag] = info["depth_median"]
                except (json.JSONDecodeError, KeyError):
                    pass

        # Read all PNGs
        png_files = sorted(
            [f for f in os.listdir(input_dir) if f.lower().endswith(".png")]
        )
        if not png_files:
            raise ValueError(f"No PNG files in: {input_dir}")

        parts_list = []
        canvas_h, canvas_w = 0, 0

        for fname in png_files:
            path = os.path.join(input_dir, fname)
            pil_img = Image.open(path).convert("RGBA")
            img_rgba = np.array(pil_img)

            # Skip fully transparent images
            if img_rgba[:, :, 3].max() < 10:
                continue

            # Upscale before decomposition
            if up_scale > 1:
                img_rgba = upscale_rgba(
                    img_rgba,
                    scale=up_scale,
                    method=up_method,
                    model_path=esrgan_model_path,
                )

            canvas_h = max(canvas_h, img_rgba.shape[0])
            canvas_w = max(canvas_w, img_rgba.shape[1])

            tag = os.path.splitext(fname)[0]

            result = decompose_layer(
                img_rgba,
                blur_size=blur_size,
                canny_low=canny_low,
                canny_high=canny_high,
                brightness_threshold=brightness_threshold,
            )

            parts_list.append(
                {
                    "tag": tag,
                    "layers": result,
                    "depth_median": depth_info.get(tag, 0.5),
                }
            )

        upscale_label = f", upscale={upscale}" if up_scale > 1 else ""
        print(
            f"[SeeThrough-Decompose] Decomposed {len(parts_list)} parts "
            f"({canvas_w}x{canvas_h}{upscale_label})"
        )

        # Build preview: composite all flat layers
        preview = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)
        sorted_parts = sorted(parts_list, key=lambda x: x["depth_median"])
        for part in sorted_parts:
            flat = part["layers"]["flat"]
            fh, fw = flat.shape[:2]
            region = preview[:fh, :fw]
            flat_alpha = flat[:, :, 3:4].astype(np.float32) / 255.0
            region[:] = (
                region.astype(np.float32) * (1 - flat_alpha)
                + flat.astype(np.float32) * flat_alpha
            ).astype(np.uint8)

        preview_tensor = (
            torch.from_numpy(preview.astype(np.float32) / 255.0).unsqueeze(0)
        )

        decomposed = DecomposedPartsData(parts_list, canvas_h, canvas_w)
        return (decomposed, preview_tensor)


# ---------------------------------------------------------------------------
# Node 3: Standalone upscale folder
# ---------------------------------------------------------------------------
class STR_UpscaleFolder:
    """
    Upscale all part PNGs in a folder and save to output folder.
    Use this to upscale See-through output before decomposition.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_dir": ("STRING", {"default": ""}),
                "output_dir": ("STRING", {"default": ""}),
                "scale": (["2", "4"], {"default": "2"}),
                "method": (["lanczos", "esrgan"], {"default": "lanczos"}),
            },
            "optional": {
                "esrgan_model_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_dir",)
    FUNCTION = "upscale_folder"
    CATEGORY = "SeeThrough-Re"

    def upscale_folder(
        self, input_dir, output_dir, scale, method, esrgan_model_path=""
    ):
        if not os.path.isdir(input_dir):
            raise ValueError(f"Directory not found: {input_dir}")

        os.makedirs(output_dir, exist_ok=True)
        scale_int = int(scale)

        png_files = sorted(
            [f for f in os.listdir(input_dir) if f.lower().endswith(".png")]
        )

        for fname in png_files:
            path = os.path.join(input_dir, fname)
            pil_img = Image.open(path).convert("RGBA")
            img_rgba = np.array(pil_img)

            if img_rgba[:, :, 3].max() < 10:
                continue

            upscaled = upscale_rgba(
                img_rgba,
                scale=scale_int,
                method=method,
                model_path=esrgan_model_path,
            )

            out_path = os.path.join(output_dir, fname)
            Image.fromarray(upscaled, "RGBA").save(out_path)

        # Copy JSON metadata if exists
        for f in os.listdir(input_dir):
            if f.endswith(".json"):
                import shutil

                shutil.copy2(
                    os.path.join(input_dir, f), os.path.join(output_dir, f)
                )

        print(
            f"[SeeThrough-Decompose] Upscaled {len(png_files)} files "
            f"({scale_int}x {method}) -> {output_dir}"
        )
        return (output_dir,)


# ---------------------------------------------------------------------------
# Node 4: Save decomposed data as PSD
# ---------------------------------------------------------------------------
class STR_SaveDecomposedPSD:
    """Save decomposed layers to a PSD file with proper blend modes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "decomposed_data": ("STR_DECOMPOSED_DATA",),
                "output_path": (
                    "STRING",
                    {"default": "output/decomposed.psd"},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("psd_path",)
    FUNCTION = "save_psd"
    OUTPUT_NODE = True
    CATEGORY = "SeeThrough-Re"

    def save_psd(self, decomposed_data, output_path):
        # Ensure output directory exists
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        save_decomposed_psd(
            decomposed_data.parts_list,
            output_path,
            decomposed_data.canvas_h,
            decomposed_data.canvas_w,
        )

        layer_count = len(decomposed_data.parts_list) * 4
        print(
            f"[SeeThrough-Decompose] Saved PSD: {output_path} "
            f"({len(decomposed_data.parts_list)} parts, {layer_count} layers)"
        )

        return (output_path,)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "STR_DecomposeLayer": STR_DecomposeLayer,
    "STR_DecomposeFolder": STR_DecomposeFolder,
    "STR_UpscaleFolder": STR_UpscaleFolder,
    "STR_SaveDecomposedPSD": STR_SaveDecomposedPSD,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "STR_DecomposeLayer": "SeeThrough-Re Decompose Layer",
    "STR_DecomposeFolder": "SeeThrough-Re Decompose Folder",
    "STR_UpscaleFolder": "SeeThrough-Re Upscale Folder",
    "STR_SaveDecomposedPSD": "SeeThrough-Re Save PSD",
}
