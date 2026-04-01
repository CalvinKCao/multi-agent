"""
Load Weights & Biases env vars from a repo-root file before wandb.init().

Looks for ``wandb.local`` (gitignored).  Same file can be shell-sourced on
Slurm: ``set -a; source wandb.local; set +a`` so one place serves Python + bash.

Search order:
  1. Path in env ``WANDB_LOCAL_FILE`` if set and the file exists
  2. ``<repo_root>/wandb.local`` where repo_root = parent of the ``drc_sokoban/`` package
  3. Walk current working directory and its parents (up to 8 levels)

Only sets ``os.environ`` keys that are **not** already set (explicit exports win).

Supported line formats (``#`` starts a comment; blank lines ignored):
  - ``WANDB_API_KEY=...``
  - ``export WANDB_API_KEY=...``
  - ``WANDB_ENTITY=...``  (optional)
  - A single non-comment line with no ``=`` is treated as the raw API key
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

_WANDB_KEYS = re.compile(r"^(WANDB_[A-Za-z0-9_]+)\s*=\s*(.*)$")
_EXPORT_PREFIX = re.compile(r"^export\s+", re.I)


def _package_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def find_wandb_local_file() -> Optional[Path]:
    override = os.environ.get("WANDB_LOCAL_FILE", "").strip()
    if override:
        p = Path(override).expanduser()
        if p.is_file():
            return p

    root = _package_repo_root()
    cand = root / "wandb.local"
    if cand.is_file():
        return cand

    here = Path.cwd().resolve()
    for d in [here, *here.parents][:9]:
        c = d / "wandb.local"
        if c.is_file():
            return c
    return None


def _strip_quotes(val: str) -> str:
    val = val.strip()
    if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
        return val[1:-1]
    return val


def load_wandb_local_env(*, verbose: bool = False) -> bool:
    """
    Parse ``wandb.local`` and apply to ``os.environ`` (missing keys only).

    Returns True if a file was found and read.
    """
    path = find_wandb_local_file()
    if path is None:
        return False

    text = path.read_text(encoding="utf-8", errors="replace")
    applied = 0
    kv_seen = False
    raw_lines: list[str] = []

    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        line = _EXPORT_PREFIX.sub("", line)
        m = _WANDB_KEYS.match(line)
        if m:
            kv_seen = True
            key, val = m.group(1), _strip_quotes(m.group(2))
            if not os.environ.get(key, "").strip():
                os.environ[key] = val
                applied += 1
        else:
            raw_lines.append(line)

    if not kv_seen and len(raw_lines) == 1 and "=" not in raw_lines[0]:
        if not os.environ.get("WANDB_API_KEY", "").strip():
            os.environ["WANDB_API_KEY"] = raw_lines[0].strip()
            applied += 1

    if verbose and applied:
        print(f"wandb_env: applied {applied} missing WANDB_* var(s) from {path}", flush=True)
    return True
