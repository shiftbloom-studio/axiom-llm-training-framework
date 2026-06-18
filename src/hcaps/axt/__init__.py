"""AXT compiler and runtime data interface for Axiom.

The public project is Axiom; `hcaps.axt` is the legacy internal package path for
the P3 structured tensor bridge.
"""

from hcaps.axt.batch import AxtBatch, AxtBatchCollator
from hcaps.axt.compiler import AxtCompileResult, compile_axt
from hcaps.axt.config import AxtCompileConfig, load_axt_compile_config
from hcaps.axt.dataset import AxtDataset, AxtRecord
from hcaps.axt.reader import AxtBundle, read_axt_bundle
from hcaps.axt.validation import validate_axt_bundle

__all__ = [
    "AxtBatch",
    "AxtBatchCollator",
    "AxtBundle",
    "AxtCompileConfig",
    "AxtCompileResult",
    "AxtDataset",
    "AxtRecord",
    "compile_axt",
    "load_axt_compile_config",
    "read_axt_bundle",
    "validate_axt_bundle",
]
