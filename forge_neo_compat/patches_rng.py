"""RNG noise-source patch.

Restores the old Forge ``get_noise_source_type`` behavior: when the active
preset is ComfyUI or DrawThings the noise source is forced to CPU at generation
time, regardless of the user's ``randn_source`` (GPU / NV). This matches the old
build exactly -- those were the only two presets it special-cased.

Important implementation detail discovered from the Forge Neo source: ``rng.py``
assigns its functions onto ``modules.devices`` at import time
(``devices.randn = randn`` etc.) and much generation code calls
``devices.randn(...)``. So we patch **both** ``modules.rng`` and
``modules.devices`` for every function that exists on each.
"""

import torch

from modules import devices, rng_philox
import modules.rng as rng_mod

from . import logging as clog
from .accessors import get_effective_noise_source
from .patch_manager import record, save_original


def _build_functions():
    """Build replacement RNG functions that resolve the noise source live.

    These mirror the current Forge Neo implementations but replace every
    ``shared.opts.randn_source`` read with :func:`get_effective_noise_source`,
    which folds in the active compatibility preset.
    """

    def compat_get_noise_source_type():
        return get_effective_noise_source()

    def manual_seed(seed):
        source = compat_get_noise_source_type()
        if source == "NV":
            rng_mod.nv_rng = rng_philox.Generator(seed)
            return
        torch.manual_seed(seed)

    def create_generator(seed):
        source = compat_get_noise_source_type()
        if source == "NV":
            return rng_philox.Generator(seed)
        device = devices.cpu if source == "CPU" or devices.device.type == "mps" else devices.device
        return torch.Generator(device).manual_seed(int(seed))

    def randn(seed, shape, generator=None):
        if generator is not None:
            manual_seed((seed + 100000) % 65536)
        else:
            manual_seed(seed)

        source = compat_get_noise_source_type()
        if source == "NV":
            return torch.asarray((generator or rng_mod.nv_rng).randn(shape), device=devices.device)
        if source == "CPU" or devices.device.type == "mps":
            return torch.randn(shape, device=devices.cpu, generator=generator).to(devices.device)
        return torch.randn(shape, device=devices.device, generator=generator)

    def randn_local(seed, shape):
        source = compat_get_noise_source_type()
        if source == "NV":
            rng = rng_philox.Generator(seed)
            return torch.asarray(rng.randn(shape), device=devices.device)
        local_device = devices.cpu if source == "CPU" or devices.device.type == "mps" else devices.device
        local_generator = torch.Generator(local_device).manual_seed(int(seed))
        return torch.randn(shape, device=local_device, generator=local_generator).to(devices.device)

    def randn_like(x):
        source = compat_get_noise_source_type()
        if source == "NV":
            return torch.asarray(rng_mod.nv_rng.randn(x.shape), device=x.device, dtype=x.dtype)
        if source == "CPU" or x.device.type == "mps":
            return torch.randn_like(x, device=devices.cpu).to(x.device)
        return torch.randn_like(x)

    def randn_without_seed(shape, generator=None):
        source = compat_get_noise_source_type()
        if source == "NV":
            return torch.asarray((generator or rng_mod.nv_rng).randn(shape), device=devices.device)
        if source == "CPU" or devices.device.type == "mps":
            return torch.randn(shape, device=devices.cpu, generator=generator).to(devices.device)
        return torch.randn(shape, device=devices.device, generator=generator)

    return {
        "get_noise_source_type": compat_get_noise_source_type,
        "manual_seed": manual_seed,
        "create_generator": create_generator,
        "randn": randn,
        "randn_local": randn_local,
        "randn_like": randn_like,
        "randn_without_seed": randn_without_seed,
    }


def install():
    functions = _build_functions()
    targets = []

    for name, fn in functions.items():
        applied_to = []
        for module in (rng_mod, devices):
            if hasattr(module, name) or name == "get_noise_source_type":
                if hasattr(module, name):
                    save_original(module, name)
                setattr(module, name, fn)
                applied_to.append(module.__name__)
        if applied_to:
            targets.append(f"{name}->{','.join(applied_to)}")

    if not targets:
        record("RNG", "skipped", "no rng/devices noise functions found")
        return

    clog.debug("RNG targets: " + "; ".join(targets))
    record("RNG", "installed", f"{len(functions)} functions on rng + devices")
