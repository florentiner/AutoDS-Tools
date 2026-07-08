"""Harbor integration for AutoDS.

This package is imported in two distinct contexts that DO NOT share a Python
environment, so keep this ``__init__`` free of heavy/optional imports:

* Host (harbor process): ``autods_harbor.agent`` / ``autods_harbor.atif`` —
  import ``harbor.*`` only.
* Container (``autods-harbor`` CLI): ``autods_harbor.entrypoint`` /
  ``autods_harbor.usage`` — import ``autods`` / langchain only.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
