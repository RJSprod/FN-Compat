"""Generation-time enforcement of the compatibility noise source.

Background
----------
The old Forge build wired the "Try to reproduce" preset directly into the live
noise path: ``modules/rng.py`` defined ``get_noise_source_type()`` and every
RNG function consulted it. Current Forge Neo deleted that helper and reads
``shared.opts.randn_source`` inline instead.

The extension's runtime monkeypatch of ``modules.rng`` aims to restore the old
behavior, but monkeypatching that module is not reliable on every build/launch
(the replacement may not be the symbol the running code actually calls). The
one value the shipping noise code is *guaranteed* to honor is
``shared.opts.randn_source`` itself.

So this Script, which Forge runs around every txt2img / img2img job, simply sets
``randn_source`` to the preset's effective source for the duration of the
generation and restores the user's value afterwards. This is non-persistent
(in-memory only; never written to ``config.json``) and reversible, mirroring the
old ``get_noise_source_type()`` "compute live, don't store" design.
"""

import os
import sys

# Make the sibling ``forge_neo_compat`` package importable regardless of the
# order in which Forge loads files under ``scripts/``.
_EXT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if _EXT_ROOT not in sys.path:
    sys.path.insert(0, _EXT_ROOT)

from modules import scripts, shared

from forge_neo_compat import logging as clog
from forge_neo_compat.accessors import (
    compat_enabled,
    get_effective_noise_source,
    option_exists,
)


class ForgeNeoCompatApply(scripts.Script):
    def __init__(self):
        super().__init__()
        # randn_source value to restore after this generation, or None if we
        # did not change it.
        self._saved_randn_source = None

    def title(self):
        return "Forge Neo Compatibility"

    def show(self, is_img2img):
        # AlwaysVisible => process()/postprocess() run for every job without
        # adding any UI controls of our own.
        return scripts.AlwaysVisible

    def process(self, p, *args):
        self._saved_randn_source = None
        try:
            if not compat_enabled():
                return
            if not option_exists("randn_source"):
                return

            effective = get_effective_noise_source()
            current = getattr(shared.opts, "randn_source", None)

            if effective and effective != current:
                self._saved_randn_source = current
                # In-memory assignment only; Options.__setattr__ does not write
                # to disk, so the user's stored preference is untouched.
                shared.opts.randn_source = effective
                clog.debug(
                    f"noise source for this job: {current!r} -> {effective!r}"
                )
        except Exception as exc:  # never let this block generation
            clog.warn(f"could not apply effective noise source: {exc}")
            self._saved_randn_source = None

    def postprocess(self, p, processed, *args):
        try:
            if self._saved_randn_source is not None:
                shared.opts.randn_source = self._saved_randn_source
                clog.debug(f"noise source restored to {self._saved_randn_source!r}")
        except Exception as exc:
            clog.warn(f"could not restore noise source: {exc}")
        finally:
            self._saved_randn_source = None
