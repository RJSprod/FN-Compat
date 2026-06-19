"""Forge Neo WebUI extension entry point.

WebUI imports everything under ``scripts/`` at startup. This thin shim puts the
extension root on ``sys.path`` so the ``forge_neo_compat`` package next to this
folder can be imported, then hands off to :func:`forge_neo_compat.bootstrap`.

All heavy lifting (settings registration + runtime patches) lives in the
package; keeping this file minimal means a syntax error here is the only thing
that could break WebUI startup, and there's almost nothing here to get wrong.
"""

import os
import sys
import traceback

PREFIX = "[Forge Neo Compatibility]"

# extensions/<this-extension>/  (parent of scripts/)
_EXT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if _EXT_ROOT not in sys.path:
    sys.path.insert(0, _EXT_ROOT)

try:
    import forge_neo_compat

    forge_neo_compat.bootstrap(basedir=_EXT_ROOT)
except Exception:
    # Never let the extension prevent WebUI from starting.
    print(f"{PREFIX} failed to load; Forge Neo will start without it:")
    traceback.print_exc()
