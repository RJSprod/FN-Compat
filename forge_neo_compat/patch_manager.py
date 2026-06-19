"""Central registry that installs, tracks, and can restore every patch.

Guarantees:
  * patches are installed at most once (unless a force-reinstall is requested);
  * a failure in one patch never blocks the others or WebUI startup;
  * every patch reports an install/skip status that feeds the debug report.
"""

import importlib
import traceback

from modules import shared

from . import logging as clog
from .accessors import compat_enabled, get_effective_noise_source, get_try_reproduce

PATCHED = False
ORIGINALS = {}
# name -> (status, detail). status is "installed" / "skipped" / "failed".
STATUS = {}


def save_original(module, attr):
    """Remember a module attribute once so it can be restored later."""
    key = f"{module.__name__}.{attr}"
    if key not in ORIGINALS:
        ORIGINALS[key] = getattr(module, attr)
    return ORIGINALS[key]


def record(name, status, detail=""):
    STATUS[name] = (status, detail)
    if status == "installed":
        clog.log(f"{name} patch installed{(': ' + detail) if detail else ''}")
    elif status == "skipped":
        clog.debug(f"{name} patch skipped: {detail}")
    else:
        clog.warn(f"{name} patch failed: {detail}")


def _run(name, fn):
    try:
        fn()
    except Exception:
        record(name, "failed", traceback.format_exc().strip().splitlines()[-1])
        clog.debug(traceback.format_exc())


def install_all_patches(*args, **kwargs):
    global PATCHED

    if not compat_enabled():
        clog.log("Compatibility runtime patches disabled by setting")
        return

    if getattr(shared.opts, "forge_neo_compat_force_patch_reinstall", False) and PATCHED:
        clog.log("Force reinstall requested; restoring originals first")
        restore_all_patches()
        PATCHED = False
        try:
            shared.opts.set("forge_neo_compat_force_patch_reinstall", False, run_callbacks=False)
            shared.opts.save(shared.config_filename)
        except Exception:
            pass

    if PATCHED:
        return

    # Imported lazily so a missing optional module can't break import order.
    from . import (
        patches_rng,
        patches_samplers,
        patches_prompt_scheduling,
        patches_hires,
        patches_refiner,
        patches_backcompat,
    )

    _run("RNG", patches_rng.install)
    _run("Karras sigmas", patches_samplers.install_karras)
    _run("DPM++ SDE batch determinism", patches_samplers.install_dpmpp_sde)
    _run("Alpha-bar downcast", patches_samplers.install_alpha_downcast)
    _run("Prompt scheduling", patches_prompt_scheduling.install)
    _run("Hires Fix", patches_hires.install)
    _run("Refiner switch", patches_refiner.install)
    _run("Auto backcompat", patches_backcompat.install)

    PATCHED = True
    print_status_report()


def restore_all_patches():
    for dotted, fn in list(ORIGINALS.items()):
        module_name, attr = dotted.rsplit(".", 1)
        try:
            module = importlib.import_module(module_name)
            setattr(module, attr, fn)
            clog.debug(f"Restored {dotted}")
        except Exception as exc:
            clog.warn(f"Could not restore {dotted}: {exc}")
    ORIGINALS.clear()
    clog.log("Original functions restored")


def print_status_report():
    from . import EXT_VERSION

    lines = [
        "================ Forge Neo Compatibility status ================",
        f"  Extension version : {EXT_VERSION}",
        f"  Master enabled    : {compat_enabled()}",
        f"  Current preset    : {get_try_reproduce()}",
        f"  Effective RNG     : {get_effective_noise_source()}",
        "  Patches:",
    ]
    for name, (status, detail) in STATUS.items():
        suffix = f" ({detail})" if detail else ""
        lines.append(f"    - {name}: {status}{suffix}")
    lines.append("================================================================")
    clog.log("\n".join(lines))
