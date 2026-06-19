"""Forge Neo Compatibility Restorer.

Restores the old Forge "Compatibility" settings page as a Forge Neo WebUI
extension and wires the controls to real runtime behavior via defensive
monkeypatching.

The package is import-safe: nothing here touches sampler internals at import
time. :func:`bootstrap` registers WebUI callbacks; the actual patches install
once the app has started.
"""

EXT_VERSION = "1.0.0"

_BOOTSTRAPPED = False


def bootstrap(basedir=None):
    """Register WebUI callbacks. Safe to call more than once."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    _BOOTSTRAPPED = True

    from . import logging as clog

    try:
        from modules import script_callbacks
    except Exception as exc:  # pragma: no cover - WebUI always provides this
        clog.warn(f"script_callbacks unavailable; extension inert: {exc}")
        return

    # Optional per-install preset overrides.
    if basedir:
        try:
            from .presets import load_external_presets
            load_external_presets(basedir)
        except Exception as exc:
            clog.warn(f"Could not load external presets: {exc}")

    from . import options, patch_manager

    script_callbacks.on_ui_settings(options.register_options)

    try:
        script_callbacks.on_app_started(patch_manager.install_all_patches)
    except Exception:
        # Some builds may not expose this callback; install immediately so the
        # core RNG behavior still works.
        patch_manager.install_all_patches()

    clog.log(f"Extension loaded (version {EXT_VERSION})")
