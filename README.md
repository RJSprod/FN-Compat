# Forge Neo Compatibility Restorer

Restores the old **Forge Compatibility** settings page as a [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)
WebUI extension — and, crucially, **wires those controls back to real generation
behavior** instead of leaving them as dead UI.

> As a Forge Neo user who previously relied on the old Forge Compatibility page,
> I want to reinstall that page as an extension so I can select presets like
> ComfyUI, Diffusers, WebUI 1.5, InvokeAI, EasyDiffusion, and DrawThings and have
> those presets actually alter generation behavior.

## What it adds

A **Settings → Stable Diffusion → Compatibility** section containing:

- **Try to reproduce the results from external software** (radio):
  `None`, `Diffusers`, `ComfyUI`, `WebUI 1.5`, `InvokeAI`, `EasyDiffusion`, `DrawThings`
- **Automatic backward compatibility**
- **Use old emphasis implementation**
- **Use old karras scheduler sigmas**
- **Do not make DPM++ SDE deterministic across different batch sizes**
- **For Hires. Fix, use Width/Height sliders to set final resolution**
- **For Hires. Fix, calculate conds of Hires. pass using Extra Networks**
- **Use old prompt editing timelines**
- **Downcast model alphas_cumprod to fp16 before sampling**
- **Switch to refiner by sampling steps instead of model timesteps**
- **Pad prompt/negative prompt to be same length** (and the legacy **v0** variant)

Plus administrative controls: `forge_neo_compat_enabled` (master switch),
`forge_neo_compat_debug_logging`, and `forge_neo_compat_force_patch_reinstall`.

## Installation

Clone into your Forge Neo `extensions/` directory and restart:

```
cd stable-diffusion-webui-forge-classic/extensions
git clone <this-repo-url> sd-forge-neo-compatibility
```

Then restart Forge Neo. You should see the Compatibility section appear and a
`[Forge Neo Compatibility]` status report printed to the console.

## How the controls take effect

| Setting | Mechanism |
|---|---|
| `forge_try_reproduce` presets | Writes lower-level options on change **and** forces the noise source live at generation time. |
| ComfyUI / DrawThings | Force noise source to **CPU** regardless of `randn_source` (GPU/NV), exactly matching old Forge's `get_noise_source_type`. Other presets defer to your `randn_source`. |
| `use_old_karras_scheduler_sigmas` | Wraps the build's `get_sigmas_karras` to clamp the sigma range to `0.1 .. 10`. |
| `use_old_scheduling` | Wraps `prompt_parser.get_learned_conditioning_prompt_schedules` to use the legacy timeline math. |
| `use_downcasted_alpha_bar` | Downcasts `alphas_cumprod` to fp16 on model load (reversibly). |
| `use_old_hires_fix_width_height`, `hires_fix_use_firstpass_conds` | Bound to the **native** Forge Neo options (no duplicate created); driven by presets. |
| `auto_backcompat` | On infotext paste, auto-enables legacy switches when old WebUI/ComfyUI/Diffusers metadata is detected. |
| `refiner_switch_by_sample_steps`, `no_dpmpp_sde_batch_determinism` | Best-effort: patched if a signature-compatible call site exists, otherwise persisted and reported as skipped. |
| `pad_cond_uncond`, `pad_cond_uncond_v0` | Recreates the gutted `CFGDenoiser.pad_cond_uncond`/`_v0` methods (Forge Neo stubs them out) and re-wires them via an `on_cfg_denoiser` callback that pads `text_cond`/`text_uncond` before sampling. |

### The key RNG detail

The old Forge build folded the preset into the live noise path via
`get_noise_source_type()` in `modules/rng.py`; current Forge Neo deleted that
helper and reads `shared.opts.randn_source` inline. The one value the shipping
noise code is guaranteed to honor is therefore `randn_source` itself.

So the primary, reliable mechanism is a generation-time `scripts.Script`
([`scripts/forge_neo_compat_apply.py`](scripts/forge_neo_compat_apply.py)) that
sets `randn_source` to the preset's effective source for the duration of each
job and restores the user's value afterwards (in-memory only; `config.json` is
never rewritten). The extension *also* monkeypatches `modules.rng` /
`modules.devices` as a secondary measure, but that is not relied upon because it
does not reliably bind on every build.

## Presets

Presets are pure data in [`forge_neo_compat/presets.py`](forge_neo_compat/presets.py).
You can override or extend them per-install without editing code by dropping a
`presets.json` at the extension root (see [`presets.json.example`](presets.json.example)):

```json
{
  "ComfyUI": { "options": { "emphasis": "Original" } },
  "MyBackend": { "runtime_force_noise_source": "CPU", "options": { "randn_source": "CPU" } }
}
```

## Safety & design

- **Never blocks startup.** The `scripts/` entry point wraps everything in
  `try/except`; individual patch failures are isolated and logged.
- **Capability-based patching.** Each patch imports its target, checks the
  function/signature exists, saves the original, installs a wrapper, and logs
  the result. Mismatched layouts are skipped with a reason — visible in the
  console status report and in debug logging.
- **No duplicate options.** Native Forge Neo options are bound, never
  re-registered under the same key.
- **Reversible.** Originals are saved and can be restored; the master switch
  disables all runtime behavior; removing the folder returns Forge Neo to
  normal (stored config values become inert).
- **No network, no external commands, no file writes outside config.**

## Disabling

- Toggle **Enable Forge Neo Compatibility Restorer** off and reload the UI, or
- Remove the extension folder and restart.

## File layout

```
sd-forge-neo-compatibility/
├─ scripts/forge_neo_compat.py        # thin WebUI entry point
├─ forge_neo_compat/
│  ├─ __init__.py                      # bootstrap(): registers callbacks
│  ├─ logging.py                       # prefixed log/debug
│  ├─ options.py                       # settings registration + apply_preset
│  ├─ presets.py                       # preset data + presets.json loader
│  ├─ accessors.py                     # canonical state accessors
│  ├─ patch_manager.py                 # install/restore/track + status report
│  ├─ patches_rng.py                   # forced-CPU noise (rng + devices)
│  ├─ patches_samplers.py              # karras sigmas, dpm++ sde, alpha-bar
│  ├─ patches_prompt_scheduling.py     # use_old_scheduling
│  ├─ patches_hires.py                 # native hires option binding
│  ├─ patches_refiner.py               # refiner-by-sampling-steps
│  ├─ patches_pad_cond.py              # pad prompt/negative prompt (cond/uncond)
│  └─ patches_backcompat.py            # auto_backcompat on infotext paste
├─ presets.json.example
└─ README.md
```

## Compatibility notes

Targets the `Haoming02/sd-webui-forge-classic` **neo** branch. Because Forge Neo
internals drift, the sampler/refiner patches are deliberately defensive: if the
expected function isn't found, the setting is still persisted and the console
status report tells you exactly which patches installed and which were skipped
and why.
