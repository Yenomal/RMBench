"""
Compatibility shim.

The codebase separates the OpenPI adapter into:
- client_adapter.py
- server_adapter.py

This file re-exports both sides for compatibility with any tooling that imports
`deploy_policy.py` directly.
"""

from .client_adapter import *  # noqa: F401,F403
from .server_adapter import *  # noqa: F401,F403
