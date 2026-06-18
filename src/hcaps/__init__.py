"""Axiom public package surface.

`hcaps` is a legacy internal package name; the public project identity is Axiom.
"""

from hcaps.schema.capsule import ClaimCapsule, HoloCapsule
from hcaps.schema.manifest import DatasetManifest

__all__ = ["ClaimCapsule", "DatasetManifest", "HoloCapsule"]

__version__ = "0.1.0"
