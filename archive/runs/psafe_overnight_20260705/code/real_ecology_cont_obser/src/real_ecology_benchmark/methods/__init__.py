"""Method registry for all seven Tier-2 policies."""

from .base import BasePolicy
from .mopo import MOPOPolicy
from .refplan import RefPlanPolicy
from .bamcts import BAMCTSPolicy
from .moor import MOORPolicy
from .plus import PLUSPolicy
from .delphic import DelphicCQLPolicy
from .ogsrl import OGSRLPolicy

METHODS = {
    "mopo": MOPOPolicy,
    "refplan": RefPlanPolicy,
    "bamcts": BAMCTSPolicy,
    "moor": MOORPolicy,
    "plus": PLUSPolicy,
    "delphic": DelphicCQLPolicy,
    "ogsrl": OGSRLPolicy,
}

__all__ = ["BasePolicy", "METHODS", *[cls.__name__ for cls in METHODS.values()]]

