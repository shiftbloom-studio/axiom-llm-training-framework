"""P6 training runtime for structured-native Axiom models."""

from hcaps.training.config import (
    ComputeBudgetConfig,
    CurriculumConfig,
    LossWeights,
    TrainingConfig,
)
from hcaps.training.loss_registry import LossRegistry, LossResult
from hcaps.training.trainer import AxiomTrainer

__all__ = [
    "AxiomTrainer",
    "ComputeBudgetConfig",
    "CurriculumConfig",
    "LossRegistry",
    "LossResult",
    "LossWeights",
    "TrainingConfig",
]
