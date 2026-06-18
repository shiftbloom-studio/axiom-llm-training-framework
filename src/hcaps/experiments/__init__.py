"""P6 experiment arm orchestration for Axiom."""

from hcaps.experiments.arms import ArmSpec, default_arm_catalog, smoke_arm_ids
from hcaps.experiments.configs import ExperimentSuiteConfig

__all__ = [
    "ArmSpec",
    "ExperimentSuiteConfig",
    "default_arm_catalog",
    "smoke_arm_ids",
]
