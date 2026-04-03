"""
Server-side adapter for serving OpenPI to RMBench.

Responsibilities:
- load OpenPI config + trained checkpoint
- keep a tiny cache compatible with RMBench's C/S protocol
- return qpos action chunks
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _ensure_openpi_import_path(openpi_root: str | os.PathLike[str]) -> None:
    root = Path(openpi_root).expanduser().resolve()
    candidates = [
        root / "src",
        root / "packages" / "openpi-client" / "src",
    ]
    for path in candidates:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


@dataclass
class OpenPIPolicyRunner:
    policy: Any
    execute_horizon: int = 1

    def __post_init__(self):
        self.obs_cache: list[dict[str, Any]] = []

    def get_obs_cache(self):
        return self.obs_cache

    def update_obs(self, obs: dict[str, Any]) -> None:
        self.obs_cache = [obs]

    def get_action(self, obs: dict[str, Any] | None = None):
        if obs is not None:
            self.update_obs(obs)
        if len(self.obs_cache) == 0:
            raise RuntimeError("Observation cache is empty.")

        result = self.policy.infer(self.obs_cache[-1])
        actions = np.asarray(result["actions"], dtype=np.float32)
        horizon = max(1, int(self.execute_horizon))
        return np.asarray(actions[:horizon], dtype=np.float32)

    def reset_model(self):
        self.obs_cache.clear()
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        return True


def get_model(usr_args):
    openpi_root = usr_args["openpi_root"]
    _ensure_openpi_import_path(openpi_root)

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    config_name = usr_args["openpi_config_name"]
    checkpoint_dir = usr_args["openpi_checkpoint_dir"]
    pytorch_device = usr_args.get("pytorch_device", None)
    default_prompt = usr_args.get("default_prompt", None)
    execute_horizon = int(usr_args.get("execute_horizon", 1))

    policy = _policy_config.create_trained_policy(
        _config.get_config(config_name),
        checkpoint_dir,
        default_prompt=default_prompt,
        pytorch_device=pytorch_device,
    )
    return OpenPIPolicyRunner(policy=policy, execute_horizon=execute_horizon)


__all__ = ["OpenPIPolicyRunner", "get_model"]
