# AI OCR Module
"""Package initialiser.

Loads a project-root ``.env`` file into the environment *before* any submodule
reads configuration at import time (e.g. ``app.auth`` captures ``DEMO_MODE`` as
a module-level constant). Python executes this file first whenever ``app.*`` is
imported, so this guarantees the values are present regardless of how the server
is launched (``uvicorn app.main:app``, ``python -m uvicorn``, tests, etc.).

Existing environment variables always win: real env (docker-compose, inline
``$env:VAR``) is never overwritten by the file.
"""

from __future__ import annotations

import os
from pathlib import Path

# .env lives in the project root, i.e. the parent of this app/ package.
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def _load_env() -> None:
    if not _ENV_PATH.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(_ENV_PATH, override=False)
        return
    except ImportError:
        pass

    # Fallback: minimal KEY=VALUE parser so the server still boots if
    # python-dotenv is not installed. Supports '#' comments, blank lines,
    # optional `export ` prefix and quoted values.
    with _ENV_PATH.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].lstrip()
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            if key:
                os.environ.setdefault(key, value)


_load_env()
