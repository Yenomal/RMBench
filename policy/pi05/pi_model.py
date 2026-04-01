from __future__ import annotations

from collections import deque
from pathlib import Path
import sys

import numpy as np


CURRENT = Path(__file__).resolve()
for parent in CURRENT.parents:
    candidate = parent / "src"
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


class PI0:
    """Thin runtime wrapper used by deploy/server adapters."""

    def __init__(
        self,
        train_config_name: str,
        model_name: str,
        checkpoint_id: str | int,
        pi0_step: int = 50,
        checkpoint_root: str | None = None,
        pytorch_device: str | None = None,
    ):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = str(checkpoint_id)
        self.pi0_step = int(pi0_step)

        policy_root = CURRENT.parent
        checkpoint_base = Path(checkpoint_root) if checkpoint_root is not None else policy_root / "checkpoints"
        self.checkpoint_dir = checkpoint_base / self.train_config_name / self.model_name / self.checkpoint_id
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {self.checkpoint_dir}")

        assets_dir = self.checkpoint_dir / "assets"
        if not assets_dir.exists():
            raise FileNotFoundError(f"Checkpoint assets directory does not exist: {assets_dir}")
        asset_entries = sorted(p for p in assets_dir.iterdir() if p.is_dir())
        if not asset_entries:
            raise FileNotFoundError(f"No asset subdirectories found under: {assets_dir}")
        asset_id = asset_entries[0].name

        config = _config.get_config(self.train_config_name)
        self.policy = _policy_config.create_trained_policy(
            config,
            str(self.checkpoint_dir),
            robotwin_repo_id=asset_id,
            pytorch_device=pytorch_device,
        )
        self.use_common_visual_encoder = getattr(config.model, "use_common_visual_encoder", False)
        self.common_visual_history_len = getattr(config.model, "common_visual_history_len", 8)
        self.head_history = deque(maxlen=self.common_visual_history_len)
        self.instruction: str | None = None
        self.observation_window: dict | None = None

    def _to_chw(self, image: np.ndarray) -> np.ndarray:
        return np.transpose(np.asarray(image, dtype=np.uint8), (2, 0, 1))

    def _update_observation_window(self, obs: dict) -> None:
        img_front = self._to_chw(obs["head_camera"])
        img_left = self._to_chw(obs["left_camera"])
        img_right = self._to_chw(obs["right_camera"])
        state = np.asarray(obs["state"], dtype=np.float32)
        instruction = "" if obs.get("instruction") is None else str(obs["instruction"])
        self.instruction = instruction

        if self.use_common_visual_encoder:
            if len(self.head_history) == 0:
                for _ in range(self.common_visual_history_len):
                    self.head_history.append(img_front.copy())
            else:
                self.head_history.append(img_front.copy())

        payload = {
            "state": state,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }
        if self.use_common_visual_encoder:
            payload["head_history"] = np.stack(list(self.head_history), axis=0)
        self.observation_window = payload

    def get_action(self, obs: dict) -> np.ndarray:
        self._update_observation_window(obs)
        assert self.observation_window is not None
        actions = self.policy.infer(self.observation_window)["actions"]
        return np.asarray(actions[: self.pi0_step], dtype=np.float32)

    def reset_model(self):
        self.instruction = None
        self.observation_window = None
        self.head_history.clear()
        return True


__all__ = ["PI0"]
