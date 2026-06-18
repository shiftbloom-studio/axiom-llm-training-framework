"""Minimal callback hooks for P6 training extensions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from hcaps.training.state import StepResult


class TrainingCallback(Protocol):
    def on_step_end(self, result: StepResult) -> None:
        """Observe a completed step."""


@dataclass(frozen=True)
class CallbackSet:
    callbacks: tuple[TrainingCallback, ...] = ()

    def on_step_end(self, result: StepResult) -> None:
        for callback in self.callbacks:
            callback.on_step_end(result)
