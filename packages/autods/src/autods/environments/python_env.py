"""Virtual environment resolution for agent subprocesses."""

from __future__ import annotations

import os
import venv
from pathlib import Path


def _system_site_packages_enabled() -> bool:
    """Whether the project venv should inherit the interpreter's site-packages.

    Enabled (via ``AUTODS_VENV_SYSTEM_SITE_PACKAGES``) in Harbor task images so
    the per-project ``.venv`` sees the specialized libraries baked into the base
    image (LightAutoML, timm, torch, ...) without reinstalling them, while the
    Coder can still ``pip install`` task-specific extras into the venv.
    """
    raw = os.getenv("AUTODS_VENV_SYSTEM_SITE_PACKAGES")
    return raw is not None and raw.strip().lower() in {"1", "true", "yes", "on"}


def _bin_subdir() -> str:
    return "Scripts" if os.name == "nt" else "bin"


def resolve_venv_env(project_path: Path) -> dict[str, str]:
    """Return env vars that make child processes use the project venv.

    In Harbor task images a pre-provisioned child venv holds the family's
    specialized libraries (LightAutoML, timm, torch, ...) isolated from the
    AutoDS framework's own interpreter. Point AutoDS's code-execution subprocesses
    at it via ``AUTODS_CHILD_VENV=/opt/venvs/<family>`` instead of creating a
    fresh per-project ``.venv``.
    """
    explicit = os.getenv("AUTODS_CHILD_VENV")
    if explicit and (Path(explicit) / _bin_subdir()).exists():
        venv_dir = Path(explicit).resolve()
    else:
        venv_dir = project_path.resolve() / ".venv"
        if not (venv_dir / _bin_subdir()).exists():
            venv.create(
                venv_dir,
                with_pip=True,
                symlinks=(os.name != "nt"),
                system_site_packages=_system_site_packages_enabled(),
            )

    bin_dir = str(venv_dir / _bin_subdir())
    env = dict(os.environ)
    env["VIRTUAL_ENV"] = str(venv_dir)
    env.pop("PYTHONHOME", None)
    path = env.get("PATH", "")
    if not path.startswith(bin_dir):
        env["PATH"] = bin_dir + os.pathsep + path
    return env
