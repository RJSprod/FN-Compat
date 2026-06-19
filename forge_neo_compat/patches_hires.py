"""Hires Fix compatibility binding.

``use_old_hires_fix_width_height`` and ``hires_fix_use_firstpass_conds`` already
ship natively in current Forge Neo (txt2img settings) and the generation code
already honors them. So there is nothing to monkeypatch: the extension simply
verifies the native options exist and lets the Compatibility page / presets
write to them.

This module exists to make that binding explicit and to log it, so the status
report is honest about why no runtime patch was applied.
"""

from .accessors import option_exists
from .patch_manager import record

_KEYS = ("use_old_hires_fix_width_height", "hires_fix_use_firstpass_conds")


def install():
    native = [k for k in _KEYS if option_exists(k)]
    missing = [k for k in _KEYS if not option_exists(k)]

    if native and not missing:
        record(
            "Hires Fix",
            "installed",
            "native options honored by Forge Neo; bound to presets (" + ", ".join(native) + ")",
        )
    elif native:
        record(
            "Hires Fix",
            "installed",
            "partial native support: " + ", ".join(native) + "; registered: " + ", ".join(missing),
        )
    else:
        # We registered them in the Compatibility section as a fallback; without
        # native generation support they persist but have no runtime effect.
        record("Hires Fix", "skipped", "no native hires options; values persisted only")
