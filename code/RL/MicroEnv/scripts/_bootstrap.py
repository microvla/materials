"""Resolve imports within a module containing its own MicroEnv source files."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Union


PathLike = Union[str, Path]


def _start_directory(start: PathLike | None) -> Path:
    """Return a resolved directory from a file or directory starting point."""
    path = Path(__file__ if start is None else start).expanduser().resolve()
    return path.parent if path.is_file() else path


def _is_repository_root(path: Path) -> bool:
    """Match a self-contained module with its bundled MicroEnv source."""
    return all(
        (
            (path / "MicroEnv" / "scripts").is_dir(),
            (path / "MicroEnv" / "MicroEnv").is_dir(),
        )
    )


def find_project_root(start: PathLike | None = None) -> Path:
    """Find and return the module root."""
    current = _start_directory(start)
    for candidate in (current, *current.parents):
        if _is_repository_root(candidate):
            return candidate

    raise RuntimeError(
        "Could not locate the module root from "
        f"{current}. Expected MicroEnv/scripts, MicroEnv/MicroEnv, "
        "under the same directory."
    )


def find_microenv_root(start: PathLike | None = None) -> Path:
    """Find and return the bundled outer MicroEnv directory."""
    return find_project_root(start) / "MicroEnv"


def _prepend_sys_path(path: Path) -> None:
    value = str(path)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)


def ensure_project_root(start: PathLike | None = None) -> Path:
    """Configure imports and return the module root.

    Both roots are added because existing scripts use both repository-qualified
    imports (``MicroEnv.MicroEnv``) and component-local imports (``vision`` and
    ``wave_generator``).  The repository root retains highest precedence.
    """
    project_root = find_project_root(start)
    microenv_root = project_root / "MicroEnv"
    _prepend_sys_path(microenv_root)
    _prepend_sys_path(project_root)
    return project_root


def ensure_microenv_root(start: PathLike | None = None) -> Path:
    """Configure imports and return the outer MicroEnv component root."""
    project_root = ensure_project_root(start)
    return project_root / "MicroEnv"


__all__ = [
    "ensure_microenv_root",
    "ensure_project_root",
    "find_microenv_root",
    "find_project_root",
]
