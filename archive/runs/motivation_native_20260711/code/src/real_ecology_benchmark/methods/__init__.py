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
}

NATIVE_METHODS = {"moor_native", "plus_native"}

__all__ = ["BasePolicy", "METHODS", *[cls.__name__ for cls in METHODS.values()]]
