"""Old prompt-editing timeline patch (``use_old_scheduling``).

Forge Neo dropped the ``use_old_scheduling`` parameter from
``get_learned_conditioning_prompt_schedules`` and always uses the modern offset
interpretation. When the legacy toggle is on we re-run the original A1111/Forge
algorithm using the live module's own ``schedule_parser`` and ``lark`` so the
edit-step math matches old behavior:

  * ``[a:b:N]`` with N < 1  -> fraction of total steps
  * ``[a:b:N]`` with N >= 1 -> absolute step number
  * Hires Fix uses the old 0..1 fractional range (no hires offset)
"""

import importlib

from . import logging as clog
from .accessors import compat_enabled, opt
from .patch_manager import record, save_original

_TARGET = "get_learned_conditioning_prompt_schedules"


def install():
    try:
        module = importlib.import_module("modules.prompt_parser")
    except Exception as exc:
        record("Prompt scheduling", "skipped", f"prompt_parser not importable: {exc}")
        return

    if not hasattr(module, _TARGET):
        record("Prompt scheduling", "skipped", f"{_TARGET} not found")
        return
    if not hasattr(module, "schedule_parser"):
        record("Prompt scheduling", "skipped", "schedule_parser not found")
        return

    original = save_original(module, _TARGET)
    setattr(module, _TARGET, _make_wrapper(module, original))
    record("Prompt scheduling", "installed", f"wrapped {_TARGET}")


def _make_wrapper(module, original):
    def wrapper(prompts, base_steps, hires_steps=None, *args, **kwargs):
        if compat_enabled() and opt("use_old_scheduling", False):
            try:
                return _old_get_schedules(module, prompts, base_steps, hires_steps)
            except Exception as exc:
                clog.warn(f"old scheduling failed, falling back to native: {exc}")
        return original(prompts, base_steps, hires_steps, *args, **kwargs)

    return wrapper


def _old_get_schedules(module, prompts, base_steps, hires_steps=None):
    """Port of the legacy ``get_learned_conditioning_prompt_schedules``.

    Uses ``use_old_scheduling=True`` semantics. Driven by the live module's own
    ``schedule_parser`` and ``lark`` to stay compatible with grammar changes.
    """
    import lark

    schedule_parser = module.schedule_parser

    # Old behavior ignores the hires offset entirely (treats it like base).
    steps = base_steps

    def collect_steps(steps, tree):
        res = [steps]

        class CollectSteps(lark.Visitor):
            def scheduled(self, tree):
                s = tree.children[-2]
                v = float(s)
                v = v * steps if v < 1 else v
                tree.children[-2] = min(steps, int(v))
                if tree.children[-2] >= 1:
                    res.append(tree.children[-2])

            def alternate(self, tree):
                res.extend(range(1, steps + 1))

        CollectSteps().visit(tree)
        return sorted(set(res))

    def at_step(step, tree):
        class AtStep(lark.Transformer):
            def scheduled(self, args):
                before, after, _, when, _ = args
                yield before or () if step <= when else after

            def alternate(self, args):
                args = ["" if not arg else arg for arg in args]
                yield args[(step - 1) % len(args)]

            def start(self, args):
                def flatten(x):
                    if isinstance(x, str):
                        yield x
                    else:
                        for gen in x:
                            yield from flatten(gen)

                return "".join(flatten(args))

            def plain(self, args):
                yield args[0].value

            def __default__(self, data, children, meta):
                for child in children:
                    yield child

        return AtStep().transform(tree)

    def get_schedule(prompt):
        try:
            tree = schedule_parser.parse(prompt)
        except lark.exceptions.LarkError:
            return [[steps, prompt]]
        return [[t, at_step(t, tree)] for t in collect_steps(steps, tree)]

    promptdict = {prompt: get_schedule(prompt) for prompt in set(prompts)}
    return [promptdict[prompt] for prompt in prompts]
