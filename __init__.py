"""ComfyUI-sageattn-ada: SageAttention kernels for consumer video DiTs on Ada.

Currently ships one node, for MiniMax H3. See README.md.
"""

from .nodes import comfy_entrypoint

__all__ = ["comfy_entrypoint"]
