"""
Text-side rebase support: adapters that let the existing CLIP-shaped methods run
on HuggingFace language models, plus ``steer_text``.

This package lives outside ``rebase/methods/`` on purpose. Importing it
registers ``steer_text`` by side effect without editing
``rebase/methods/__init__.py``, so every file under ``methods/`` stays untouched.
"""

from __future__ import annotations

from .adapters import (
    TextEncoderShim,
    alias_inputs_loader,
    attach_local_labels,
    balanced_indices,
    count_transformer_blocks,
    describe_key_coverage,
    head_intermediate_linears,
    head_linear,
    subset_loader,
    text_param_filter,
)
from .steer_text import (  # noqa: F401  -- import registers "steer_text"
    SteerTextRebase,
    block_index_of,
    block_modules,
    num_residual_blocks,
    steer_text_correction_context,
    text_parameter_blocks,
)

__all__ = [
    "SteerTextRebase",
    "TextEncoderShim",
    "alias_inputs_loader",
    "attach_local_labels",
    "balanced_indices",
    "block_index_of",
    "block_modules",
    "count_transformer_blocks",
    "describe_key_coverage",
    "head_intermediate_linears",
    "head_linear",
    "num_residual_blocks",
    "steer_text_correction_context",
    "subset_loader",
    "text_param_filter",
    "text_parameter_blocks",
]
