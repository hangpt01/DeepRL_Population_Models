"""Method registry for all seven benchmark policies."""

from .base import BasePolicy
from .mopo import MOPOPolicy
from .refplan import RefPlanPolicy
from .bamcts import BAMCTSPolicy
from .moor import MOORPolicy
from .plus import PLUSPolicy
from .moor_native import MOORNativePolicy
from .plus_native import PLUSNativePolicy
from .ensemble_value_disagreement import EnsembleValueDisagreementPolicy
from .ogsrl import OGSRLPolicy
from .moor_faithful import MOORFaithfulRickerPBVIPolicy
from .plus_faithful import PLUSFaithfulPBVIPolicy

METHODS = {
    "mopo": MOPOPolicy,
    "refplan": RefPlanPolicy,
    "bamcts": BAMCTSPolicy,
    "moor": MOORPolicy,
    "plus": PLUSPolicy,
    "moor_native": MOORNativePolicy,
    "plus_native": PLUSNativePolicy,
    "ensemble_value_disagreement_pessimism": EnsembleValueDisagreementPolicy,
    # Deprecated read/CLI alias for historical artifacts only.
    "delphic": EnsembleValueDisagreementPolicy,
    "ogsrl": OGSRLPolicy,
    "moor_adapted_ricker_misspec_pbvi": MOORFaithfulRickerPBVIPolicy,
    "plus_adapted_mechanistic_pbvi": PLUSFaithfulPBVIPolicy,
}

METHOD_LABELS = {
    "refplan": "RefPlan-inspired",
    "ogsrl": "OGSRL-inspired",
    "bamcts": "BA-MCTS-inspired",
    "ensemble_value_disagreement_pessimism": (
        "Ensemble value-disagreement pessimism (Delphic-motivated)"
    ),
    "delphic": "DEPRECATED alias: Ensemble value-disagreement pessimism (Delphic-motivated)",
}

NATIVE_METHODS = {"moor_native", "plus_native"}
FAITHFUL_METHODS = {
    "moor_adapted_ricker_misspec_pbvi",
    "plus_adapted_mechanistic_pbvi",
}

__all__ = [
    "BasePolicy", "METHODS", "METHOD_LABELS", "NATIVE_METHODS", "FAITHFUL_METHODS",
    *[cls.__name__ for cls in METHODS.values()],
]
