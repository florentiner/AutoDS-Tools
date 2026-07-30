"""Shared constants for the AutoDS Harbor integration (stdlib-only).

Imported from both the host-side agent and the container-side entrypoint, so
this module must not import ``harbor`` or ``autods``.
"""

from __future__ import annotations

# Raw AutoDS trace the entrypoint writes inside the container (agent log dir).
# The host-side agent reads it back and converts it to ATIF.
RAW_TRACE_FILENAME = "autods_trace.json"
# Human-readable console log tee'd during the run.
AGENT_LOG_FILENAME = "autods.log"
# Canonical ATIF file Harbor discovers in the agent log dir.
TRAJECTORY_FILENAME = "trajectory.json"

# In-container workspace where task data is staged and the submission is written.
DEFAULT_WORKSPACE = "/workspace"
# Instruction file uploaded into the container.
DEFAULT_INSTRUCTION_PATH = "/installed-agent/instruction.md"
# Where the agent installs AutoDS in task images that don't bake it in (the
# general path for arbitrary / registry datasets). Also used as the child venv.
INSTALL_VENV = "/opt/autods-agent-venv"
# Container path for the generated pip requirements when installing AutoDS.
INSTALL_REQUIREMENTS_PATH = "/tmp/autods-install-req.txt"

# Env vars forwarded from the harbor host process into the agent container so
# AutoDS can reach its OpenAI-compatible LLM gateway and be tuned per run.
FORWARDED_ENV_VARS = (
    "AUTODS_MODEL",
    "AUTODS_API_KEY",
    "AUTODS_BASE_URL",
    "AUTODS_MAX_RETRIES",
    "AUTODS_MODEL_KWARGS_JSON",
    "AUTODS_EXTRA_BODY_JSON",
    "AUTODS_DEFAULT_HEADERS_JSON",
    "AUTODS_PRICE_INPUT_PER_1M",
    "AUTODS_PRICE_OUTPUT_PER_1M",
    # LangGraph super-step budget for long runs.
    "AUTODS_RECURSION_LIMIT",
    # Pipeline toggles.
    "RESEARCH_DISABLED",
    "LIBQ_DISABLED",
    "DEBUGGER_DISABLED",
    # Make the per-project venv inherit the image's baked specialized libs.
    "AUTODS_VENV_SYSTEM_SITE_PACKAGES",
    # Where task code runs (a pre-provisioned specialized-lib venv).
    "AUTODS_CHILD_VENV",
    # pip/git spec(s) to install AutoDS into non-baked task images (branch 2).
    "AUTODS_INSTALL_SPEC",
)
