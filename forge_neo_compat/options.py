"""Settings registration for the restored Compatibility page.

Adds a ``Settings -> Stable Diffusion -> Compatibility`` section. Native Forge
Neo options (such as ``use_old_hires_fix_width_height``) are never re-registered
under the same key; presets simply write to them where they already exist.
"""

import gradio as gr

from modules import shared
from modules.options import OptionInfo

from . import logging as clog
from .accessors import compat_enabled, get_try_reproduce, option_exists
from .presets import PRESET_CHOICES, PRESETS

# Forge Neo's ``ui_settings.create_ui`` unpacks ``item.section`` as a 2-tuple
# (``elem_id, text = item.section``), so the section MUST be exactly
# ``(section_id, section_label)``. The settings *category* is a separate concept
# applied via ``OptionInfo.category_id`` -- ``"sd"`` places this under the
# existing "Stable Diffusion" settings category, matching old Forge. Passing a
# 3-tuple as the section crashes UI creation with
# "ValueError: too many values to unpack (expected 2)".
SECTION = ("compatibility", "Compatibility")
CATEGORY_ID = "sd"


def _opt(*args, **kwargs):
    """Build an :class:`OptionInfo` bound to our section and category.

    The category is applied as an attribute *after* construction so we don't
    depend on the host build's ``OptionInfo`` accepting a ``category_id``
    keyword. On builds that have no category concept this is a harmless no-op.
    """
    info = OptionInfo(*args, section=SECTION, **kwargs)
    try:
        info.category_id = CATEGORY_ID
    except Exception as exc:  # pragma: no cover - defensive only
        clog.debug(f"Could not set category_id on option: {exc}")
    return info

# Keys owned (created) by this extension. Used by the status report.
OWNED_KEYS = []

# Legacy toggles that already ship natively in some Forge Neo builds. For these
# we bind to the existing option instead of creating a duplicate.
LEGACY_TOGGLES = [
    ("use_old_emphasis_implementation", False,
     "Use old emphasis implementation. Can be useful to reproduce old seeds."),
    ("use_old_karras_scheduler_sigmas", False,
     "Use old karras scheduler sigmas (0.1 to 10)."),
    ("no_dpmpp_sde_batch_determinism", False,
     "Do not make DPM++ SDE deterministic across different batch sizes."),
    ("use_old_hires_fix_width_height", False,
     "For Hires. Fix, use Width/Height sliders to set final resolution rather than first pass resolution."),
    ("hires_fix_use_firstpass_conds", False,
     "For Hires. Fix, calculate conds of Hires. Fix pass using extra networks of first pass."),
    ("use_old_scheduling", False,
     "Use old prompt editing timelines."),
    ("use_downcasted_alpha_bar", False,
     "Downcast model alphas_cumprod to fp16 before sampling."),
    ("refiner_switch_by_sample_steps", False,
     "Switch to refiner by sampling steps instead of model timesteps."),
]


def add_option_if_missing(name, option_info):
    if option_exists(name):
        clog.debug(f"Option already exists; binding instead of re-adding: {name}")
        return False
    shared.opts.add_option(name, option_info)
    OWNED_KEYS.append(name)
    clog.debug(f"Option added: {name}")
    return True


def apply_preset():
    """onchange handler for ``forge_try_reproduce``.

    Writes the selected preset's lower-level options into ``shared.opts`` (only
    keys that exist), then persists. Runtime patches additionally read the
    preset live via :func:`accessors.get_effective_noise_source`.
    """
    if not compat_enabled():
        clog.debug("apply_preset skipped: compatibility disabled")
        return

    selected = get_try_reproduce()
    preset = PRESETS.get(selected, PRESETS["None"])
    for key, value in preset.get("options", {}).items():
        if not option_exists(key):
            clog.debug(f"Preset {selected}: skipping unknown option {key}")
            continue
        try:
            shared.opts.set(key, value, run_callbacks=False)
            clog.debug(f"Preset {selected}: set {key}={value!r}")
        except Exception as exc:
            clog.warn(f"Preset {selected}: could not set {key}: {exc}")

    try:
        shared.opts.save(shared.config_filename)
    except Exception as exc:
        clog.warn(f"Could not save config after applying preset: {exc}")

    clog.log(f"Applied preset: {selected}")


def _on_old_emphasis_changed():
    """Map the legacy emphasis toggle to the closest modern equivalent.

    Forge Neo routes emphasis through ``sd_emphasis`` rather than the old
    weighting code, so the faithful approximation is to pin emphasis to
    "Original" while the legacy flag is on.
    """
    if not compat_enabled():
        return
    if not option_exists("emphasis"):
        return
    if getattr(shared.opts, "use_old_emphasis_implementation", False):
        try:
            shared.opts.set("emphasis", "Original", run_callbacks=False)
            shared.opts.save(shared.config_filename)
            clog.debug("use_old_emphasis_implementation: emphasis pinned to Original")
        except Exception as exc:
            clog.warn(f"Could not pin emphasis to Original: {exc}")


def register_options():
    """Registered via ``script_callbacks.on_ui_settings``."""

    # --- Administrative controls -------------------------------------------
    add_option_if_missing(
        "forge_neo_compat_enabled",
        _opt(
            True,
            "Enable Forge Neo Compatibility Restorer (master switch for runtime patches)",
        ).needs_reload_ui(),
    )
    add_option_if_missing(
        "forge_neo_compat_debug_logging",
        _opt(
            False,
            "Compatibility Restorer debug logging (prints patch decisions to console)",
        ),
    )
    add_option_if_missing(
        "forge_neo_compat_force_patch_reinstall",
        _opt(
            False,
            "Reinstall runtime patches on next UI reload",
        ),
    )

    # --- Try to reproduce results from external software -------------------
    add_option_if_missing(
        "forge_try_reproduce",
        _opt(
            "None",
            "Try to reproduce the results from external software",
            gr.Radio,
            {"choices": list(PRESET_CHOICES)},
            onchange=apply_preset,
        ).info("Selecting a preset rewrites the lower-level options below and "
               "forces the noise source at generation time."),
    )

    # --- Automatic backward compatibility ----------------------------------
    add_option_if_missing(
        "auto_backcompat",
        _opt(
            True,
            "Automatic backward compatibility",
        ).info("Auto-enable relevant legacy switches based on imported infotext/version metadata."),
    )

    # --- Legacy toggles -----------------------------------------------------
    for key, default, label in LEGACY_TOGGLES:
        if key == "use_old_emphasis_implementation":
            info = _opt(default, label, onchange=_on_old_emphasis_changed)
        else:
            info = _opt(default, label)
        add_option_if_missing(key, info)

    # --- Pad prompt / negative prompt --------------------------------------
    # Recreated from old Forge/A1111; Forge Neo gutted these. The runtime patch
    # in patches_pad_cond reads these keys live.
    add_option_if_missing(
        "pad_cond_uncond",
        _opt(
            False,
            "Pad prompt/negative prompt to be same length",
        ).info("Pads the shorter of the positive/negative conditioning so both "
               "match; helps when prompt and negative prompt differ in length. "
               "Changes seeds."),
    )
    add_option_if_missing(
        "pad_cond_uncond_v0",
        _opt(
            False,
            "Pad prompt/negative prompt to be same length (v0)",
        ).info("Legacy v0 padding behavior (repeat last vector / truncate). "
               "Takes precedence over the non-v0 option. Changes seeds."),
    )

    clog.log("Options registered")
