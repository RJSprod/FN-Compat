"""Automatic backward compatibility (``auto_backcompat``).

When the user pastes/loads generation parameters (infotext or PNG metadata)
that carry an old program/version marker, relevant legacy switches are
auto-enabled before the parameters are applied.

Hooked via ``script_callbacks.on_infotext_pasted`` which fires with
``(infotext, parsed_params_dict)``.
"""

import re

from modules import shared

from . import logging as clog
from .accessors import compat_enabled, option_exists

_WEBUI_VER_RE = re.compile(r"v?(\d+)\.(\d+)")


def _set(key, value):
    if option_exists(key):
        try:
            shared.opts.set(key, value, run_callbacks=False)
            clog.debug(f"auto_backcompat: set {key}={value!r}")
            return True
        except Exception as exc:
            clog.warn(f"auto_backcompat: could not set {key}: {exc}")
    return False


def _on_infotext_pasted(infotext, params):
    if not compat_enabled():
        return
    if not bool(getattr(shared.opts, "auto_backcompat", True)):
        return

    params = params or {}
    changed = False

    # Gather candidate version/software markers from common keys.
    blob = " ".join(
        str(params.get(k, ""))
        for k in ("Version", "Program Version", "Software", "App", "Source")
    )
    blob_l = blob.lower()
    full = (str(infotext or "") + " " + blob).lower()

    # Old WebUI (<= 1.5) style metadata -> enable the 1.5-era legacy switches.
    is_old_webui = False
    m = _WEBUI_VER_RE.search(blob)
    if m:
        major, minor = int(m.group(1)), int(m.group(2))
        if (major, minor) <= (1, 5):
            is_old_webui = True
    if is_old_webui:
        for key in (
            "use_old_scheduling",
            "use_downcasted_alpha_bar",
            "use_old_hires_fix_width_height",
        ):
            changed |= _set(key, True)

    # External-program markers -> set the reproduce preset.
    if "comfyui" in full or "comfy" in blob_l:
        changed |= _set("forge_try_reproduce", "ComfyUI")
    elif "diffusers" in full:
        changed |= _set("forge_try_reproduce", "Diffusers")
    elif "invokeai" in full or "invoke-ai" in full:
        changed |= _set("forge_try_reproduce", "InvokeAI")

    if changed:
        try:
            shared.opts.save(shared.config_filename)
        except Exception:
            pass
        clog.log("auto_backcompat applied legacy switches from imported metadata")


def install():
    from modules import script_callbacks

    script_callbacks.on_infotext_pasted(_on_infotext_pasted)
    clog.debug("auto_backcompat: on_infotext_pasted hook registered")
