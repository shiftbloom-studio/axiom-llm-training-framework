"""P6 curriculum schedules that change weights without changing data validity."""

from __future__ import annotations

from hcaps.training.config import CurriculumConfig

RELATION_PHASE = 2
GEOMETRY_PHASE = 4


def curriculum_multipliers(
    config: CurriculumConfig, *, step: int, max_steps: int
) -> dict[str, float]:
    """Return loss/exposure multipliers for a step.

    The curriculum never turns target-only fields into predictor inputs; it only
    scales loss families and declared control exposure.
    """

    if config.schedule_name == "none":
        return {
            "text_projection": 1.0,
            "structured": 1.0,
            "relations": 1.0,
            "provenance": 1.0,
            "geometry": 1.0,
        }
    progress = float(step + 1) / float(max(max_steps, 1))
    if config.schedule_name == "linear_complexity_ramp":
        structured = max(0.1, progress)
        return {
            "text_projection": 1.0,
            "structured": structured,
            "relations": structured,
            "provenance": structured,
            "geometry": progress,
        }
    phase = _phase_index(config.phase_boundaries, step)
    return {
        "text_projection": 1.0,
        "structured": 1.0 if phase >= 0 else 0.0,
        "relations": 1.0 if phase >= RELATION_PHASE else 0.0,
        "provenance": 1.0 if phase >= RELATION_PHASE else 0.0,
        "geometry": 1.0 if phase >= GEOMETRY_PHASE else 0.0,
    }


def _phase_index(boundaries: list[int], step: int) -> int:
    phase = 0
    for index, boundary in enumerate(boundaries):
        if step >= boundary:
            phase = index
    return phase
