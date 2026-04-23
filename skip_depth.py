"""
SeeThrough_SkipDepth — GenerateDepthをスキップするブリッジノード

GenerateLayers の出力 (SEETHROUGH_LAYERS) を受け取り、
ダミーの深度マップを生成して SEETHROUGH_LAYERS_DEPTH に変換する。

これにより GenerateLayers → SkipDepth → PostProcess → SavePSD の
深度推定なしパイプラインが可能になる。

インストール:
  このファイルを ComfyUI/custom_nodes/ に配置して ComfyUI を再起動。

使い方:
  GenerateLayers → SeeThrough_SkipDepth → PostProcess → SavePSD
  (GenerateDepth と LoadDepthModel は不要)
"""

import numpy as np

# Import the data classes from See-through nodes
import sys
import os

# Try to import from the See-through node
try:
    seethrough_dir = os.path.join(os.path.dirname(__file__), "ComfyUI-See-through")
    if not os.path.exists(seethrough_dir):
        # Check in sibling custom_nodes directory
        custom_nodes_dir = os.path.dirname(__file__)
        seethrough_dir = os.path.join(custom_nodes_dir, "ComfyUI-See-through")

    # We need the data classes - define them locally to avoid import issues
    class SeeThrough_LayersData:
        def __init__(self, layer_dict, fullpage, input_img, resolution, pad_size, pad_pos):
            self.layer_dict = layer_dict
            self.fullpage = fullpage
            self.input_img = input_img
            self.resolution = resolution
            self.pad_size = pad_size
            self.pad_pos = pad_pos
            self.scale = pad_size[0] / resolution

    class SeeThrough_LayersDepthData:
        def __init__(self, layer_dict, depth_dict, fullpage, resolution):
            self.layer_dict = layer_dict
            self.depth_dict = depth_dict
            self.fullpage = fullpage
            self.resolution = resolution

except Exception as e:
    print(f"[SkipDepth] Warning: {e}")


class SeeThrough_SkipDepth:
    """Skip depth estimation by generating flat dummy depth maps.

    Assigns depth values based on a predefined layer ordering so that
    PostProcess can still produce correctly ordered PSD layers.
    """

    # Predefined depth order (front → back)
    DEPTH_ORDER = [
        "front hair", "eyelash", "irides", "eyewhite", "eyebrow",
        "mouth", "nose", "face", "headwear", "ears", "earwear",
        "eyewear", "neck", "neckwear", "topwear", "handwear",
        "bottomwear", "legwear", "footwear", "objects",
        "tail", "wings", "back hair", "head",
    ]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "layers": ("SEETHROUGH_LAYERS",),
            },
        }

    RETURN_TYPES = ("SEETHROUGH_LAYERS_DEPTH", "IMAGE")
    RETURN_NAMES = ("layers_depth", "preview")
    FUNCTION = "skip_depth"
    CATEGORY = "SeeThrough"

    def skip_depth(self, layers):
        import torch

        layer_dict = layers.layer_dict
        fullpage = layers.fullpage
        resolution = layers.resolution

        print(f"[SeeThrough] SkipDepth: generating dummy depth for {len(layer_dict)} layers", flush=True)

        # Generate dummy depth maps based on predefined ordering
        depth_dict = {}
        tags = list(layer_dict.keys())

        for tag in tags:
            img = layer_dict[tag]
            h, w = img.shape[:2]

            # Assign depth based on predefined order
            if tag in self.DEPTH_ORDER:
                idx = self.DEPTH_ORDER.index(tag)
                depth_val = idx / max(len(self.DEPTH_ORDER) - 1, 1)
            else:
                depth_val = 0.5  # Unknown tags get middle depth

            # Create flat depth map
            depth = np.full((h, w), depth_val, dtype=np.float32)
            depth_dict[tag] = depth

        # Create LayersDepthData
        layers_depth = SeeThrough_LayersDepthData(
            layer_dict=layer_dict,
            depth_dict=depth_dict,
            fullpage=fullpage,
            resolution=resolution,
        )

        # Generate preview (same as GenerateDepth would)
        preview = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        for tag in tags:
            img = layer_dict[tag]
            mask = img[..., -1] > 10
            depth_val = depth_dict[tag]
            if np.any(mask):
                gray = (depth_val[mask] * 255).astype(np.uint8)
                preview[mask, 0] = gray
                preview[mask, 1] = gray
                preview[mask, 2] = gray

        preview_tensor = torch.from_numpy(preview).float() / 255.0
        preview_tensor = preview_tensor.unsqueeze(0)

        print(f"[SeeThrough] SkipDepth complete: {len(depth_dict)} dummy depth maps", flush=True)
        return (layers_depth, preview_tensor)


NODE_CLASS_MAPPINGS = {
    "SeeThrough_SkipDepth": SeeThrough_SkipDepth,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SeeThrough_SkipDepth": "SeeThrough Skip Depth (深度マップスキップ)",
}
