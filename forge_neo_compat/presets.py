"""Preset definitions for the ``forge_try_reproduce`` setting.

Presets are pure data so a maintainer (or an end user, via ``presets.json``)
can tune the mapping in one place without touching any patch code.

Each preset entry may contain:

    runtime_force_noise_source:
        "CPU" / "GPU" / "NV" / None. Read at generation time by the RNG patch
        to override ``shared.opts.randn_source`` without permanently rewriting
        the user's stored value.

    options:
        A flat mapping of ``shared.opts`` keys to values that are written when
        the preset is selected. Only keys that actually exist in the running
        Forge Neo build are applied; everything else is skipped and logged.

        NOTE: deliberately do *not* put ``randn_source`` here. The noise source
        is forced live and reversibly via ``runtime_force_noise_source`` (read
        at generation time) so the user's stored ``randn_source`` is never
        clobbered. Writing ``randn_source`` into ``options`` would persist it
        and -- because deselecting a preset cannot know the user's prior value
        -- leave every later generation stuck on that source.
"""

import json
import os

from . import logging as clog

PRESET_CHOICES = [
    "None",
    "Diffusers",
    "ComfyUI",
    "WebUI 1.5",
    "InvokeAI",
    "EasyDiffusion",
    "DrawThings",
]

# Presets that force the noise source to CPU at runtime. This mirrors the old
# Forge ``get_noise_source_type`` *exactly*: only ComfyUI and DrawThings were
# special-cased to CPU; every other preset deferred to the user's randn_source.
CPU_NOISE_PRESETS = {"ComfyUI", "DrawThings"}

PRESETS = {
    "None": {
        "runtime_force_noise_source": None,
        "options": {},
    },
    "ComfyUI": {
        "runtime_force_noise_source": "CPU",
        "options": {
            "emphasis": "Original",
            "sdxl_crop_top": 0,
            "sdxl_crop_left": 0,
        },
    },
    "DrawThings": {
        "runtime_force_noise_source": "CPU",
        "options": {
            "emphasis": "Original",
        },
    },
    "Diffusers": {
        # Old Forge did not force the noise source for Diffusers; it only
        # adjusted the settings below and left randn_source to the user.
        "runtime_force_noise_source": None,
        "options": {
            "emphasis": "Original",
            "sdxl_crop_top": 0,
            "sdxl_crop_left": 0,
        },
    },
    "WebUI 1.5": {
        "runtime_force_noise_source": None,
        "options": {
            "emphasis": "Original",
            "use_old_hires_fix_width_height": True,
            "hires_fix_use_firstpass_conds": True,
            "use_old_scheduling": True,
            "use_downcasted_alpha_bar": True,
        },
    },
    "InvokeAI": {
        # Old Forge did not force the noise source for InvokeAI.
        "runtime_force_noise_source": None,
        "options": {
            "emphasis": "Original",
            "sdxl_crop_top": 0,
            "sdxl_crop_left": 0,
        },
    },
    "EasyDiffusion": {
        # Old Forge did not force the noise source for EasyDiffusion.
        "runtime_force_noise_source": None,
        "options": {
            "emphasis": "Original",
        },
    },
}


def load_external_presets(basedir):
    """Merge an optional ``presets.json`` sitting at the extension root.

    The JSON file may add new presets or override fields of built-in ones. It is
    never required; a malformed file is logged and ignored so it can't break
    startup.
    """
    path = os.path.join(basedir, "presets.json")
    if not os.path.isfile(path):
        return

    try:
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
    except Exception as exc:
        clog.warn(f"Could not read presets.json: {exc}")
        return

    if not isinstance(data, dict):
        clog.warn("presets.json must contain a JSON object; ignoring")
        return

    for name, entry in data.items():
        if not isinstance(entry, dict):
            continue
        merged = dict(PRESETS.get(name, {"runtime_force_noise_source": None, "options": {}}))
        if "runtime_force_noise_source" in entry:
            merged["runtime_force_noise_source"] = entry["runtime_force_noise_source"]
        if isinstance(entry.get("options"), dict):
            opts = dict(merged.get("options", {}))
            opts.update(entry["options"])
            merged["options"] = opts
        PRESETS[name] = merged
        if name not in PRESET_CHOICES:
            PRESET_CHOICES.append(name)
        clog.debug(f"Loaded external preset override: {name}")
