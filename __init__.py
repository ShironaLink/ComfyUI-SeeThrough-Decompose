"""
ComfyUI-SeeThrough-Decompose
Layer decomposition pipeline for See-through output.

Pipeline: See-through parts -> Base/Highlight/Shadow split -> Canny lineart -> PSD
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
