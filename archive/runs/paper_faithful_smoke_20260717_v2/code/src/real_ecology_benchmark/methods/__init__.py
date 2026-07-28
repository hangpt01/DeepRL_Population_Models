"""Method registry for all seven benchmark policies."""

from .base import BasePolicy
from .mopo import MOPOPolicy
from .refplan import RefPlanPolicy
from .bamcts import BAMCTSPolicy
from .moor import MOORPolicy
from .plus import PLUSPolicy
from .moor_native import MOORNativePolicy
from .plus_native import PLUSNativePolicy
from .delphic import DelphicCQLPolicy
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
    "delphic": DelphicCQLPolicy,
    "ogsrl": OGSRLPolicy,
    "moor_faithful_ricker_misspec_pbvi": MOORFaithfulRickerPBVIPolicy,
    "plus_faithful_pbvi": PLUSFaithfulPBVIPolicy,
}

NATIVE_METHODS = {"moor_native", "plus_native"}
FAITHFUL_METHODS = {
    "moor_faithful_ricker_misspec_pbvi",
    "plus_faithful_pbvi",
}

__all__ = [
    "BasePolicy", "METHODS", "NATIVE_METHODS", "FAITHFUL_METHODS",
    *[cls.__name__ for cls in METHODS.values()],
]
