"""Canonical Axiom claim-state schema models.

`HoloCapsule` remains as a legacy internal model alias for compatibility.
"""

from hcaps.schema.capsule import (
    Claim,
    ClaimCapsule,
    ContextFiber,
    EpistemicState,
    HoloCapsule,
    LineageRecord,
    ProvenanceRecord,
    QualitySignals,
    Relation,
    SurfaceForms,
    TrainingTargets,
)
from hcaps.schema.document import DocumentSpan, SourceDocument, TemporalCutoff
from hcaps.schema.manifest import DatasetManifest

__all__ = [
    "Claim",
    "ClaimCapsule",
    "ContextFiber",
    "DatasetManifest",
    "DocumentSpan",
    "EpistemicState",
    "HoloCapsule",
    "LineageRecord",
    "ProvenanceRecord",
    "QualitySignals",
    "Relation",
    "SourceDocument",
    "SurfaceForms",
    "TemporalCutoff",
    "TrainingTargets",
]
