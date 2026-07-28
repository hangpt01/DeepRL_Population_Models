"""Common planner interface and provenance payload."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np

from ..faithful_pomdp import CandidateBelief, CandidatePOMDP


@dataclass(frozen=True)
class PlannerProvenance:
    name: str
    implementation_version: str
    seed: int
    model_hash: str
    invocation_count: int
    external_invocation: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class BeliefPlanner(Protocol):
    model: CandidatePOMDP

    def action_values(self, belief: CandidateBelief) -> np.ndarray:
        ...

    def provenance(self) -> PlannerProvenance:
        ...
