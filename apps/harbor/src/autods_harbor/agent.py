"""Harbor installed-agent adapter for AutoDS.

Runs the multi-agent AutoDS pipeline *inside* the task container via the
``autods-harbor`` entrypoint (baked into the task base image), then converts the
AutoDS trace into a Harbor ATIF ``trajectory.json`` so the hub renders per-step
reasoning, tool calls, tokens and cost like any first-class agent.

This module runs in the harbor host process and imports ``harbor`` only — never
``autods`` (which lives in the container). See ``__init__`` for the split.
"""

from __future__ import annotations

import shlex
from pathlib import PurePosixPath
from typing import Any, override

from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths

from autods_harbor import c1_prompt
from autods_harbor.constants import (
    AGENT_LOG_FILENAME,
    DEFAULT_INSTRUCTION_PATH,
    DEFAULT_WORKSPACE,
    FORWARDED_ENV_VARS,
    INSTALL_REQUIREMENTS_PATH,
    INSTALL_VENV,
    RAW_TRACE_FILENAME,
    TRAJECTORY_FILENAME,
)


class AutoDSAgent(BaseInstalledAgent):
    """Run the AutoDS pipeline as a Harbor agent (native ATIF traces)."""

    SUPPORTS_ATIF: bool = True

    _REMOTE_INSTRUCTION_PATH = PurePosixPath(DEFAULT_INSTRUCTION_PATH)

    def __init__(
        self,
        workspace: str = DEFAULT_WORKSPACE,
        install_spec: str | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.workspace = workspace
        # pip requirement(s) used to install AutoDS into task images that do not
        # bake it in (branch 2 / arbitrary Harbor datasets). Newline- or
        # space-separated; also settable via --ae AUTODS_INSTALL_SPEC=...
        self.install_spec = install_spec
        # Set when the general install path provisions a venv; then task code
        # runs there (AUTODS_CHILD_VENV) instead of the baked family venv.
        self._installed_child_venv: str | None = None

    @staticmethod
    @override
    def name() -> str:
        return "autods"

    @override
    def get_version_command(self) -> str | None:
        return "autods-harbor --help >/dev/null 2>&1 && echo autods-harbor"

    async def _autods_present(self, environment: BaseEnvironment) -> bool:
        try:
            await self.exec_as_agent(environment, command="command -v autods-harbor >/dev/null 2>&1")
            return True
        except Exception:  # noqa: BLE001 - absence is expected on non-baked images
            return False

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        # Fast path: AutoDS + autods-harbor are baked into the task base image
        # (autods-mlab-*, branch 1). Nothing to do.
        if await self._autods_present(environment):
            return

        # General path (branch 2 / registry datasets): install AutoDS into the
        # container from a pip/git spec so the agent works on ANY task image.
        spec = self._get_env("AUTODS_INSTALL_SPEC") or self.install_spec
        if not spec:
            raise RuntimeError(
                "autods-harbor is not baked into this task image and no install "
                "spec was provided. Either build the task FROM an autods-mlab-* "
                "base image, or pass the AutoDS source as a pip spec, e.g.:\n"
                "  --ak install_spec='git+https://github.com/<you>/AutoDS-Tools.git#subdirectory=packages/autods "
                "git+https://github.com/<you>/AutoDS-Tools.git#subdirectory=apps/harbor'"
            )
        await self._install_autods(environment, spec)
        self._installed_child_venv = INSTALL_VENV

    async def _install_autods(self, environment: BaseEnvironment, spec: str) -> None:
        # Ensure curl (to fetch uv) using whichever package manager exists.
        await self.exec_as_root(
            environment,
            command=(
                "if ! command -v curl >/dev/null 2>&1; then "
                "  (apt-get update && apt-get install -y curl) "
                "  || apk add --no-cache curl "
                "  || (yum install -y curl) || true; "
                "fi"
            ),
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        await self.exec_as_root(
            environment,
            command=f"mkdir -p {shlex.quote(INSTALL_VENV)} && chown -R $(id -u):$(id -g) {shlex.quote(INSTALL_VENV)}",
        )
        # Build a uv-managed py3.12 venv and install AutoDS + autods-harbor. The
        # spec is passed via env (never interpolated into the shell) so URL '#'
        # fragments aren't treated as comments; written to a requirements file.
        install = (
            "set -e; "
            "curl -LsSf https://astral.sh/uv/install.sh | sh; "
            '. "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"; '
            "uv python install 3.12; "
            f"uv venv {shlex.quote(INSTALL_VENV)} --python 3.12 --seed; "
            f"printf '%s\\n' \"$AUTODS_INSTALL_SPEC\" > {shlex.quote(INSTALL_REQUIREMENTS_PATH)}; "
            f"{shlex.quote(INSTALL_VENV + '/bin/pip')} install -r {shlex.quote(INSTALL_REQUIREMENTS_PATH)}"
        )
        await self.exec_as_agent(environment, command=install, env={"AUTODS_INSTALL_SPEC": spec})
        # Put autods-harbor on PATH for the run() exec.
        await self.exec_as_root(
            environment,
            command=f"ln -sf {shlex.quote(INSTALL_VENV + '/bin/autods-harbor')} /usr/local/bin/autods-harbor",
        )

    def _agent_env(self) -> dict[str, str]:
        """Env vars handed to the in-container AutoDS run."""
        env: dict[str, str] = {}
        for key in FORWARDED_ENV_VARS:
            value = self._get_env(key)  # extra_env (from --ae) wins over os.environ
            if value is not None and value != "":
                env[key] = value
        # Map Harbor's -m/--model onto AUTODS_MODEL when not set explicitly.
        if "AUTODS_MODEL" not in env and self.model_name:
            env["AUTODS_MODEL"] = self.model_name
        # When AutoDS was installed by the agent (non-baked image), run task code
        # in that venv; otherwise let the per-project venv inherit the baked libs.
        if self._installed_child_venv and "AUTODS_CHILD_VENV" not in env:
            env["AUTODS_CHILD_VENV"] = self._installed_child_venv
        env.setdefault("AUTODS_VENV_SYSTEM_SITE_PACKAGES", "1")
        return env

    async def _detect_family(self, environment: BaseEnvironment) -> str:
        """Best-effort task modality from the staged workspace (AUTODS_FAMILY wins).

        Used only to select the AutoDS C1 (specialized-library) layer; the task's
        own instruction stays modality-neutral for every other agent.
        """
        override = self._get_env("AUTODS_FAMILY")
        if override:
            return c1_prompt.normalize_family(override)
        try:
            res = await self.exec_as_agent(
                environment,
                command=(
                    "ls -1 /workspace 2>/dev/null; echo '<<<H>>>'; "
                    "head -1 /workspace/train.csv 2>/dev/null"
                ),
            )
            out = (getattr(res, "stdout", "") or "").lower()
        except Exception:  # noqa: BLE001 - detection is best-effort
            return "tabular"
        files, _, header = out.partition("<<<h>>>")
        if "node_features.npy" in files or "edges.npy" in files:
            return "graph"
        if "_images.npy" in files:
            return "vision"
        if "text" in [c.strip() for c in header.split(",")]:
            return "nlp"
        return "tabular"

    @with_prompt_template
    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        # AutoDS-only C1 layer: the task ships the ORIGINAL benchmark instruction;
        # here we append the modality-specific specialized-library + training
        # discipline blocks so only AutoDS sees them. Skip with AUTODS_C1_DISABLED=1.
        if not self._get_env("AUTODS_C1_DISABLED"):
            family = await self._detect_family(environment)
            instruction = c1_prompt.augment(instruction, family)
            self.logger.info("AutoDS C1 layer applied (family=%s)", family)

        instruction_path = self.logs_dir / "instruction.md"
        instruction_path.write_text(instruction, encoding="utf-8")
        await environment.upload_file(instruction_path, self._REMOTE_INSTRUCTION_PATH.as_posix())

        agent_dir = EnvironmentPaths.agent_dir
        trace_out = (agent_dir / RAW_TRACE_FILENAME).as_posix()
        log_out = (agent_dir / AGENT_LOG_FILENAME).as_posix()
        command = (
            "autods-harbor "
            f"--instruction-file {shlex.quote(self._REMOTE_INSTRUCTION_PATH.as_posix())} "
            f"--workspace {shlex.quote(self.workspace)} "
            f"--trace-out {shlex.quote(trace_out)} "
            f"2>&1 | stdbuf -oL tee {shlex.quote(log_out)}"
        )
        await self.exec_as_agent(environment, command=command, env=self._agent_env(), cwd=self.workspace)
        # NOTE: deliberately do NOT populate `context` here. Harbor only calls
        # populate_context_post_run() when the agent context is still empty
        # (trial.py `_populate_agent_context`), so mutating context in run() would
        # skip ATIF trajectory generation. All backfill happens below.

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        """Convert the synced AutoDS trace into ATIF and backfill usage/metadata."""
        context.metadata = {
            **(context.metadata or {}),
            "autods_workspace": self.workspace,
            "autods_model": self.model_name,
        }
        raw_trace = self.logs_dir / RAW_TRACE_FILENAME
        if not raw_trace.exists():
            self.logger.debug("No AutoDS raw trace at %s; skipping ATIF", raw_trace)
            return
        try:
            # Imported lazily: harbor.models is host-only and avoids importing at
            # module load in case the package is present in a slim environment.
            from autods_harbor.atif import convert_trace_file

            totals = convert_trace_file(raw_trace, self.logs_dir / TRAJECTORY_FILENAME)
        except Exception as exc:  # noqa: BLE001 - trajectory is best-effort
            self.logger.warning("Failed to convert AutoDS trace to ATIF: %s", exc)
            return

        if totals.n_input_tokens is not None:
            context.n_input_tokens = totals.n_input_tokens
        if totals.n_output_tokens is not None:
            context.n_output_tokens = totals.n_output_tokens
        if totals.n_cache_tokens is not None:
            context.n_cache_tokens = totals.n_cache_tokens
        if totals.cost_usd is not None:
            context.cost_usd = totals.cost_usd


__all__ = ["AutoDSAgent"]
