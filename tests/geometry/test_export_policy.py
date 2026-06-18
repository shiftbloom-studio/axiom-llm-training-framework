from __future__ import annotations

import orjson
import torch

from hcaps.geometry.config import GeometryMode
from hcaps.geometry.export import geometry_observables_to_axc_out_fields
from hcaps.geometry.observables import empty_observables


def test_axc_out_geometry_fields_are_json_safe_and_forbid_truth() -> None:
    payload = geometry_observables_to_axc_out_fields(
        mode=GeometryMode.LEARNED,
        enabled=True,
        observables=empty_observables(device=torch.device("cpu")),
    )
    encoded = orjson.dumps(payload)
    assert encoded
    lowered = encoded.decode("utf-8").lower()
    assert "truth" not in lowered
    assert "raw_connection" not in lowered
