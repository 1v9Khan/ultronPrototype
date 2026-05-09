"""V1-spec gap C2: backwards-compat alias for ``dump_session.py``.

V1 prompt Part 6.4 asked for a ``scripts/last_session.py`` script that
renders the most recent coding-session audit log into a readable
transcript. The actual implementation landed under the more
descriptive name ``dump_session.py``. This file is a one-line alias
so users following the original spec wording still find the
expected entry point.

Both names share a single source of truth -- editing
``scripts/dump_session.py`` updates this alias automatically.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_SOURCE = Path(__file__).resolve().parent / "dump_session.py"


def main(argv=None) -> int:
    """Forward to :func:`scripts.dump_session.main`.

    Imports the source module by file path so the alias works whether
    or not ``scripts/`` is on ``sys.path`` as a package.
    """
    spec = importlib.util.spec_from_file_location(
        "_dump_session_aliased", _SOURCE,
    )
    if spec is None or spec.loader is None:
        print(f"ERROR: could not load {_SOURCE}", file=sys.stderr)
        return 2
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main(argv if argv is not None else sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
