"""Refiner switch-by-sampling-steps patch (``refiner_switch_by_sample_steps``).

Old Forge could hand off to the refiner based on the sampling step index rather
than the model timestep. Forge Neo decides the refiner handoff inside its
processing/sampler path, and the exact decision function is not stable across
builds.

We look for a recognizable refiner-switch decision point and wrap it; if none is
found we report the patch as skipped with a reason (the setting is still
persisted and surfaced), per the spec's "active or explicitly skipped" rule.
"""

import importlib
import inspect

from . import logging as clog
from .accessors import compat_enabled, opt
from .patch_manager import record, save_original

_CANDIDATE_MODULES = (
    "modules.processing",
    "modules.sd_samplers_common",
    "modules.sd_samplers_kdiffusion",
)
_NAME_HINTS = ("refiner",)
_SWITCH_HINTS = ("switch", "should")


def install():
    for module_name in _CANDIDATE_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        for attr in dir(module):
            low = attr.lower()
            if not any(h in low for h in _NAME_HINTS):
                continue
            if not any(h in low for h in _SWITCH_HINTS):
                continue
            target = getattr(module, attr)
            if not callable(target) or inspect.isclass(target):
                continue

            try:
                sig = inspect.signature(target)
            except (TypeError, ValueError):
                continue

            params = set(sig.parameters)
            # We need both a current-step and a total-steps notion to recompute
            # the handoff by sampling step.
            step_param = _first_match(params, ("current_step", "step", "i"))
            total_param = _first_match(params, ("total_steps", "steps", "n"))
            switch_param = _first_match(params, ("switch_at", "refiner_switch_at", "at"))
            if not (step_param and total_param and switch_param):
                continue

            original = save_original(module, attr)
            setattr(
                module,
                attr,
                _make_wrapper(original, sig, step_param, total_param, switch_param),
            )
            record("Refiner switch", "installed", f"{module_name}.{attr}")
            return

    record(
        "Refiner switch",
        "skipped",
        "no signature-compatible refiner-switch function found; setting persisted only",
    )


def _first_match(params, names):
    for name in names:
        if name in params:
            return name
    return None


def _make_wrapper(fn, sig, step_param, total_param, switch_param):
    def wrapper(*args, **kwargs):
        if compat_enabled() and opt("refiner_switch_by_sample_steps", False):
            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                current = bound.arguments.get(step_param)
                total = bound.arguments.get(total_param)
                switch_at = bound.arguments.get(switch_param)
                if current is not None and total and switch_at is not None:
                    result = int(current) >= int(int(total) * float(switch_at))
                    clog.debug(
                        f"refiner-by-step: step {current}/{total} switch_at {switch_at} -> {result}"
                    )
                    return result
            except Exception as exc:
                clog.debug(f"refiner-by-step wrapper fell back to original: {exc}")
        return fn(*args, **kwargs)

    return wrapper
