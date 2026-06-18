from __future__ import annotations

from pathlib import Path

from hcaps.falsification.audits import audit_future_target_exposure
from hcaps.falsification.runner import load_capsules
from hcaps.training_bridge.examples import (
    render_capsule_text,
    render_flat_text,
    render_structured_text,
)

EXAMPLES = Path("examples/axf/v0_1")


def test_predictor_renderers_do_not_expose_future_target_fields() -> None:
    capsules, _, _ = load_capsules(EXAMPLES / "minimal_capsules.axc")
    rendered = []
    for capsule in capsules:
        rendered.extend(
            [
                render_flat_text(capsule),
                render_structured_text(capsule),
                render_capsule_text(capsule),
            ]
        )

    assert audit_future_target_exposure(rendered) == []
    assert "future_summary" not in "\n".join(rendered)
    assert "target_timestamp" not in "\n".join(rendered)
