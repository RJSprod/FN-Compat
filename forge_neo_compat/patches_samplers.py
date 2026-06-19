"""Sampler / scheduler compatibility patches.

Covers three legacy switches:
  * use_old_karras_scheduler_sigmas  (capability-based wrap of get_sigmas_karras)
  * no_dpmpp_sde_batch_determinism   (best-effort; reported if not locatable)
  * use_downcasted_alpha_bar         (model alphas_cumprod downcast on load)

All discovery is defensive: if the current Forge Neo layout doesn't match an
expected signature the patch is skipped with a logged reason rather than
guessing destructively.
"""

import importlib
import inspect

import torch

from modules import script_callbacks

from . import logging as clog
from .accessors import compat_enabled, opt
from .patch_manager import record, save_original

_KARRAS_CANDIDATE_MODULES = (
    "modules.sd_schedulers",
    "modules.sd_samplers_kdiffusion",
    "modules.sd_samplers_common",
)


def install_karras():
    for module_name in _KARRAS_CANDIDATE_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        for attr in dir(module):
            low = attr.lower()
            if "karras" not in low or "sigma" not in low:
                continue
            target = getattr(module, attr)
            if not callable(target):
                continue

            original = save_original(module, attr)
            try:
                sig = inspect.signature(original)
            except (TypeError, ValueError):
                sig = None

            setattr(module, attr, _make_karras_wrapper(original, sig))
            record("Karras sigmas", "installed", f"{module_name}.{attr}")
            return

    record("Karras sigmas", "skipped", "no get_sigmas_karras-like target found")


def _make_karras_wrapper(fn, sig):
    def wrapper(*args, **kwargs):
        if compat_enabled() and opt("use_old_karras_scheduler_sigmas", False):
            # Old Forge/WebUI used a fixed 0.1..10 sigma range for Karras.
            if sig is not None:
                try:
                    bound = sig.bind(*args, **kwargs)
                    if "sigma_min" in sig.parameters:
                        bound.arguments["sigma_min"] = 0.1
                    if "sigma_max" in sig.parameters:
                        bound.arguments["sigma_max"] = 10.0
                    return fn(*bound.args, **bound.kwargs)
                except TypeError:
                    pass
            else:
                if "sigma_min" in kwargs:
                    kwargs["sigma_min"] = 0.1
                if "sigma_max" in kwargs:
                    kwargs["sigma_max"] = 10.0
        return fn(*args, **kwargs)

    return wrapper


def install_dpmpp_sde():
    """DPM++ SDE batch determinism.

    Forge Neo's k-diffusion sampling does not expose a stable, signature-safe
    hook for the batch-determinism correction, so we report this as skipped with
    a reason instead of patching blindly. The setting is still persisted and
    surfaced; a maintainer can wire it to the exact call site once identified
    for a given build.
    """
    try:
        importlib.import_module("modules.sd_samplers_kdiffusion")
    except Exception:
        record("DPM++ SDE batch determinism", "skipped", "sd_samplers_kdiffusion not importable")
        return
    record(
        "DPM++ SDE batch determinism",
        "skipped",
        "no stable hook in this build; setting persisted but not actively patched",
    )


# ---------------------------------------------------------------------------
# alphas_cumprod downcast
# ---------------------------------------------------------------------------

_ATTR = "_forge_neo_compat_orig_alphas_cumprod"


def _find_alphas_holder(model):
    """Return (object, attr_name) that owns an ``alphas_cumprod`` tensor."""
    for holder in (model, getattr(model, "model", None), getattr(model, "forge_objects", None)):
        if holder is not None and hasattr(holder, "alphas_cumprod"):
            return holder, "alphas_cumprod"
    return None, None


def _apply_alpha_downcast(model):
    if model is None:
        return
    holder, attr = _find_alphas_holder(model)
    if holder is None:
        clog.debug("alpha-bar: no alphas_cumprod tensor found on model")
        return

    want = compat_enabled() and opt("use_downcasted_alpha_bar", False)
    current = getattr(holder, attr)

    if want:
        if not hasattr(holder, _ATTR):
            try:
                setattr(holder, _ATTR, (current, current.dtype))
            except Exception:
                pass
        try:
            if current.dtype != torch.float16:
                setattr(holder, attr, current.half())
                clog.debug("alpha-bar: downcast alphas_cumprod to fp16")
        except Exception as exc:
            clog.warn(f"alpha-bar: downcast failed: {exc}")
    else:
        # Restore the original tensor if we previously downcast it.
        saved = getattr(holder, _ATTR, None)
        if saved is not None:
            try:
                setattr(holder, attr, saved[0].to(saved[1]))
                clog.debug("alpha-bar: restored original alphas_cumprod")
            except Exception:
                pass


def install_alpha_downcast():
    # Re-evaluate on every model load so toggling the setting + reloading the
    # model takes effect, and so we never irreversibly clobber a cached model.
    script_callbacks.on_model_loaded(_apply_alpha_downcast)
    record("Alpha-bar downcast", "installed", "hooked on_model_loaded")
