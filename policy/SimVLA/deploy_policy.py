"""
Compatibility shim.

The codebase now separates the old shared adapter into:
- client_adapter.py
- server_adapter.py

This file re-exports both sides for compatibility with any older tooling that
still imports `deploy_policy.py` directly.
"""

from .client_adapter import *  # noqa: F401,F403
from .server_adapter import *  # noqa: F401,F403
