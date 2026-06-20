"""Pad prompt / negative prompt patch (``pad_cond_uncond`` + ``pad_cond_uncond_v0``).

Old Forge (and A1111) exposed *"Pad prompt/negative prompt to be same length"* on
the Compatibility page. It padded the shorter of the positive/negative
conditioning tensors so both have the same sequence length, which changes seeds
and lets the two be processed together.

Forge Neo deliberately **gutted** this: ``CFGDenoiser.pad_cond_uncond`` and
``pad_cond_uncond_v0`` are stubs that ``raise NotImplementedError`` and there is
no longer any option driving them. This patch restores the feature in two parts:

1. **Recreate the methods.** We monkeypatch real implementations back onto the
   ``CFGDenoiser`` class (faithful ports of the A1111 originals, dict-aware for
   SDXL), replacing the ``NotImplementedError`` stubs.
2. **Re-wire them into generation.** Forge Neo's ``forward()`` no longer calls
   those methods, so we register an ``on_cfg_denoiser`` callback. It fires right
   before ``sampling_function`` consumes ``CFGDenoiserParams.text_cond`` /
   ``text_uncond``, so padding the tensors there flows straight into sampling.

The callback is gated by the master switch and the per-feature options, so the
patch is effectively reversible by toggling the setting off (mirroring the
alpha-bar downcast patch, which is likewise callback-based).
"""

import importlib

import torch

from modules import script_callbacks

from . import logging as clog
from .accessors import compat_enabled, opt
from .patch_manager import record

_MODULE = "modules.sd_samplers_cfg_denoiser"

# Originals are kept so the recreated methods can be reverted in place; the
# callback is gated by the option, so disabling the setting is the primary
# (and always-available) "off switch".
_ORIGINAL_METHODS = {}


def _is_dict_cond(tensor):
    # ``DictWithShape`` (SDXL: {"crossattn", "vector"}) subclasses ``dict``.
    return isinstance(tensor, dict)


def _crossattn(tensor):
    return tensor["crossattn"] if _is_dict_cond(tensor) else tensor


def _seq_len(tensor):
    """Sequence-token length (dim 1) of a cond tensor or DictWithShape."""
    return _crossattn(tensor).shape[1]


def _pad_cond(tensor, repeats, empty):
    """Append ``repeats`` copies of ``empty`` along the token axis (dim 1).

    Mirrors ``modules.sd_samplers_cfg_denoiser.pad_cond``: for dict conds only
    the ``crossattn`` stream is padded; the pooled ``vector`` is left untouched.
    """
    if not _is_dict_cond(tensor):
        return torch.cat([tensor, empty.repeat((tensor.shape[0], repeats, 1))], axis=1)
    tensor["crossattn"] = _pad_cond(tensor["crossattn"], repeats, empty)
    return tensor


def _get_empty_prompt_embedding():
    """The empty-prompt embedding A1111 pads with, or ``None`` if unavailable.

    Forge Neo's backend does not always expose ``cond_stage_model_empty_prompt``;
    when it is missing the caller falls back to repeating the shorter tensor's
    own last token vector (the same idiom ``prompt_parser.stack_conds`` uses for
    length mismatches), so the feature still works on those builds.
    """
    try:
        from modules import shared

        empty = getattr(shared.sd_model, "cond_stage_model_empty_prompt", None)
        if isinstance(empty, torch.Tensor):
            return empty
    except Exception as exc:  # pragma: no cover - defensive only
        clog.debug(f"pad cond: empty-prompt embedding lookup failed: {exc}")
    return None


def _extend_by_last_vector(tensor, target_len):
    """Pad ``tensor`` up to ``target_len`` tokens by repeating its last vector."""
    vec = _crossattn(tensor)
    pad = target_len - vec.shape[1]
    if pad <= 0:
        return tensor
    last = vec[:, -1:]
    extended = torch.cat([vec, last.repeat([1, pad, 1])], axis=1)
    if _is_dict_cond(tensor):
        tensor["crossattn"] = extended
        return tensor
    return extended


def pad_cond_uncond(self, cond, uncond):
    """Port of A1111's ``CFGDenoiser.pad_cond_uncond``.

    Pads the shorter of cond/uncond up to the longer using the empty-prompt
    embedding (in whole chunks, as A1111 did); falls back to repeating the
    shorter tensor's last vector when no empty-prompt embedding is available.
    """
    empty = _get_empty_prompt_embedding()
    cond_len = _seq_len(cond)
    uncond_len = _seq_len(uncond)

    if empty is not None and empty.shape[1] > 0:
        num_repeats = (cond_len - uncond_len) // empty.shape[1]
        if num_repeats < 0:
            cond = _pad_cond(cond, -num_repeats, empty)
            self.padded_cond_uncond = True
        elif num_repeats > 0:
            uncond = _pad_cond(uncond, num_repeats, empty)
            self.padded_cond_uncond = True
        return cond, uncond

    # Fallback: extend whichever is shorter, never truncate.
    if uncond_len < cond_len:
        uncond = _extend_by_last_vector(uncond, cond_len)
        self.padded_cond_uncond = True
    elif cond_len < uncond_len:
        cond = _extend_by_last_vector(cond, uncond_len)
        self.padded_cond_uncond = True
    return cond, uncond


def pad_cond_uncond_v0(self, cond, uncond):
    """Port of A1111's legacy ``CFGDenoiser.pad_cond_uncond_v0``.

    The v0 behaviour is intentionally asymmetric: when uncond is shorter it is
    extended by repeating its last vector; when uncond is longer the positive
    cond is *truncated* to match. Preserved verbatim for old-seed reproduction.
    """
    is_dict_cond = _is_dict_cond(uncond)
    uncond_vec = uncond["crossattn"] if is_dict_cond else uncond
    cond_vec = cond["crossattn"] if is_dict_cond else cond

    if uncond_vec.shape[1] < cond_vec.shape[1]:
        last_vector = uncond_vec[:, -1:]
        last_vector_repeated = last_vector.repeat([1, cond_vec.shape[1] - uncond_vec.shape[1], 1])
        uncond_vec = torch.hstack([uncond_vec, last_vector_repeated])
        self.padded_cond_uncond_v0 = True
    elif uncond_vec.shape[1] > cond_vec.shape[1]:
        cond_vec = cond_vec[:, : uncond_vec.shape[1]]
        self.padded_cond_uncond_v0 = True

    if is_dict_cond:
        uncond["crossattn"] = uncond_vec
        cond["crossattn"] = cond_vec
    else:
        uncond = uncond_vec
        cond = cond_vec

    return cond, uncond


def _record_infotext(denoiser, key):
    """Surface the active padding mode in the generation parameters."""
    try:
        params = denoiser.p.extra_generation_params
        params[key] = True
    except Exception:
        pass


def _on_cfg_denoiser(params):
    """``on_cfg_denoiser`` handler: pad text_cond/text_uncond before sampling."""
    if not compat_enabled():
        return

    use_v0 = bool(opt("pad_cond_uncond_v0", False))
    use_main = bool(opt("pad_cond_uncond", False))
    if not (use_main or use_v0):
        return

    cond = getattr(params, "text_cond", None)
    uncond = getattr(params, "text_uncond", None)
    if cond is None or uncond is None:
        return

    try:
        if _seq_len(cond) == _seq_len(uncond):
            return

        denoiser = getattr(params, "denoiser", None)
        if denoiser is None:
            return

        if use_v0:
            cond, uncond = denoiser.pad_cond_uncond_v0(cond, uncond)
            _record_infotext(denoiser, "Pad conds v0")
        else:
            cond, uncond = denoiser.pad_cond_uncond(cond, uncond)
            _record_infotext(denoiser, "Pad conds")

        params.text_cond = cond
        params.text_uncond = uncond
        clog.debug(
            f"pad cond/uncond ({'v0' if use_v0 else 'main'}): "
            f"cond={_seq_len(cond)} uncond={_seq_len(uncond)}"
        )
    except Exception as exc:  # never break generation
        clog.warn(f"pad cond/uncond failed, leaving conds unchanged: {exc}")


def install():
    try:
        module = importlib.import_module(_MODULE)
    except Exception as exc:
        record("Pad cond/uncond", "skipped", f"{_MODULE} not importable: {exc}")
        return

    cls = getattr(module, "CFGDenoiser", None)
    if cls is None:
        record("Pad cond/uncond", "skipped", "CFGDenoiser class not found")
        return

    # Recreate the gutted methods (the literal monkeypatch). Replacing a
    # NotImplementedError stub is non-destructive even while the option is off,
    # since nothing invokes the methods unless the callback below does.
    for name, impl in (("pad_cond_uncond", pad_cond_uncond), ("pad_cond_uncond_v0", pad_cond_uncond_v0)):
        if name not in _ORIGINAL_METHODS:
            _ORIGINAL_METHODS[name] = getattr(cls, name, None)
        setattr(cls, name, impl)

    if not hasattr(script_callbacks, "on_cfg_denoiser"):
        record(
            "Pad cond/uncond",
            "skipped",
            "methods recreated but on_cfg_denoiser callback unavailable; not wired into sampling",
        )
        return

    script_callbacks.on_cfg_denoiser(_on_cfg_denoiser)
    record("Pad cond/uncond", "installed", "recreated CFGDenoiser pad methods + on_cfg_denoiser hook")


def restore():
    """Best-effort restoration of the original stub methods."""
    try:
        module = importlib.import_module(_MODULE)
        cls = getattr(module, "CFGDenoiser", None)
        if cls is None:
            return
        for name, original in _ORIGINAL_METHODS.items():
            if original is not None:
                setattr(cls, name, original)
        clog.debug("Restored original CFGDenoiser pad methods")
    except Exception as exc:  # pragma: no cover - defensive only
        clog.warn(f"Could not restore CFGDenoiser pad methods: {exc}")
