"""Tiny logger wrapper for the Forge Neo Compatibility extension.

Kept dependency-free so it can be imported at any point during startup, even
before ``shared.opts`` has been populated.
"""

PREFIX = "[Forge Neo Compatibility]"


def _opts():
    try:
        from modules import shared
        return getattr(shared, "opts", None)
    except Exception:
        return None


def log(message):
    print(f"{PREFIX} {message}")


def debug(message):
    opts = _opts()
    if opts is not None and getattr(opts, "forge_neo_compat_debug_logging", False):
        log("DEBUG: " + str(message))


def warn(message):
    log("WARNING: " + str(message))
