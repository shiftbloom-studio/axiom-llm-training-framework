"""Structured-native Axiom model stack.

`hcaps` is the legacy internal package name. Publicly this package implements the
Axiom P4 structured-native LLM model stack: AXT in, native AXC-out plus text
projection out.
"""

from hcaps.model.axc_out import AXCOutEmission, AXCOutRawEmission
from hcaps.model.config import AxiomModelConfig
from hcaps.model.factory import AxiomStructuredModel
from hcaps.model.input_adapter import StructuredInputAdapter
from hcaps.model.serialization import load_model_checkpoint, save_model_checkpoint
from hcaps.model.types import AxiomModelInput, AxiomModelOutput, TypedSlotBundle

__all__ = [
    "AXCOutEmission",
    "AXCOutRawEmission",
    "AxiomModelConfig",
    "AxiomModelInput",
    "AxiomModelOutput",
    "AxiomStructuredModel",
    "StructuredInputAdapter",
    "TypedSlotBundle",
    "load_model_checkpoint",
    "save_model_checkpoint",
]
