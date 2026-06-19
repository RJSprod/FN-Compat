"""Canonical accessors that shield runtime patches from key/name drift.

Every patch reads compatibility state through these helpers rather than poking
``shared.opts`` directly, so that a single change here (e.g. a renamed key)
keeps all patches consistent.
"""

from modules import shared

from .presets import PRESETS


def _opts():
    return getattr(shared, "opts", None)


def option_exists(name):
    opts = _opts()
    return opts is not None and name in getattr(opts, "data_labels", {})


def get_try_reproduce():
    """Return the active preset name from the canonical key, with fallbacks."""
    opts = _opts()
    if opts is None:
        return "None"
    # Prefer the old Forge key; fall back to the namespaced alternative.
    for key in ("forge_try_reproduce", "neo_forge_try_reproduce"):
        if key in getattr(opts, "data_labels", {}):
            value = getattr(opts, key, None)
            if value:
                return value
    return "None"


def compat_enabled():
    """Master switch for all runtime monkeypatch behavior."""
    opts = _opts()
    if opts is None:
        return True
    return bool(getattr(opts, "forge_neo_compat_enabled", True))


def get_randn_source():
    opts = _opts()
    if opts is None:
        return "GPU"
    return getattr(opts, "randn_source", "GPU")


def get_effective_noise_source():
    """The noise source that should actually be used for this generation.

    Returns the preset's forced source (e.g. CPU for ComfyUI/DrawThings) when a
    preset demands it, otherwise the user's ``randn_source``.
    """
    base = get_randn_source()
    if not compat_enabled():
        return base
    forced = PRESETS.get(get_try_reproduce(), {}).get("runtime_force_noise_source")
    return forced or base


def opt(name, default=None):
    opts = _opts()
    if opts is None:
        return default
    return getattr(opts, name, default)
