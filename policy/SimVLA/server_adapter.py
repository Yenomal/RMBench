"""
Server-side adapter for serving SimVLA to RMBench.

Responsibilities:
- load SimVLA + processor + norm stats
- convert request payload into model inputs
- return action chunks
"""

from __future__ import annotations

import json
import os
import sys
from collections import deque
from typing import Dict, Iterable

import numpy as np


DEFAULT_CAMERA_KEYS = ("head_camera", "left_camera", "right_camera")


def _simvla_root() -> str:
    """Resolve the local SimVLA package root."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "SimVLA"))


def _load_checkpoint_config(checkpoint_path: str):
    """Load a local HF-style config.json without using from_pretrained."""
    config_path = os.path.join(checkpoint_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config.json under checkpoint path: {checkpoint_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_checkpoint_state_dict(checkpoint_path: str, torch_module):
    """Load a local HF-style state dict from safetensors or pytorch_model.bin."""
    safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    pytorch_bin_path = os.path.join(checkpoint_path, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file

        return load_file(safetensors_path, device="cpu")

    if os.path.exists(pytorch_bin_path):
        return torch_module.load(pytorch_bin_path, map_location="cpu")

    raise FileNotFoundError(
        f"Missing checkpoint weights under {checkpoint_path}; expected model.safetensors or pytorch_model.bin"
    )


class SimVLAPolicyRunner:
    """
    Thin runtime wrapper around SimVLA for RMBench inference.
    """

    def __init__(
        self,
        checkpoint_path: str,
        smolvlm_model_path: str | None = None,
        norm_stats_path: str | None = None,
        device: str = "cuda",
        execute_horizon: int = 5,
        integration_steps: int = 10,
        camera_keys: Iterable[str] = DEFAULT_CAMERA_KEYS,
        expected_state_dim: int = 14,
    ):
        simvla_root = _simvla_root()
        if simvla_root not in sys.path:
            sys.path.insert(0, simvla_root)

        import torch
        from models.configuration_smolvlm_vla import SmolVLMVLAConfig
        from models.modeling_smolvlm_vla import SmolVLMVLA
        from models.processing_smolvlm_vla import SmolVLMVLAProcessor

        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.camera_keys = list(camera_keys)
        self.execute_horizon = max(1, int(execute_horizon))
        self.integration_steps = max(1, int(integration_steps))
        self.expected_state_dim = int(expected_state_dim)
        self.obs_cache = deque(maxlen=1)

        config_dict = _load_checkpoint_config(checkpoint_path)
        if smolvlm_model_path:
            config_dict["smolvlm_model_path"] = smolvlm_model_path

        config = SmolVLMVLAConfig(**config_dict)
        self.model = SmolVLMVLA(config)
        state_dict = _load_checkpoint_state_dict(checkpoint_path, torch)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            raise RuntimeError(f"Missing keys when loading checkpoint {checkpoint_path}: {missing[:20]}")
        if unexpected:
            raise RuntimeError(f"Unexpected keys when loading checkpoint {checkpoint_path}: {unexpected[:20]}")

        self.model = self.model.to(self.device)
        self.model.eval()

        processor_path = smolvlm_model_path or getattr(self.model.config, "smolvlm_model_path", checkpoint_path)
        self.processor = SmolVLMVLAProcessor.from_pretrained(processor_path)

        if norm_stats_path:
            self.model.action_space.load_norm_stats(norm_stats_path)

    def reset_model(self):
        """Reset any local policy cache."""
        self.obs_cache.clear()
        return True

    def _prepare_inputs(self, obs: Dict[str, object]) -> Dict[str, object]:
        """Convert raw numpy observation into model inputs."""
        images = [obs[k] for k in self.camera_keys if k in obs]
        if len(images) == 0:
            raise ValueError("At least one camera image is required")

        state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
        if state.shape[0] != self.expected_state_dim:
            raise ValueError(
                f"Expected state dim {self.expected_state_dim}, got {state.shape[0]}. "
                "This baseline currently targets the default aloha-agilex embodiment."
            )

        image_inputs = self.processor.encode_image(images)
        language_inputs = self.processor.encode_language(obs.get("instruction", ""))

        def to_device(t):
            if not isinstance(t, self.torch.Tensor):
                t = self.torch.as_tensor(t)
            if t.is_floating_point():
                return t.to(device=self.device, dtype=self.torch.float32)
            return t.to(device=self.device)

        inputs = {k: to_device(v) for k, v in image_inputs.items()}
        inputs.update({k: to_device(v) for k, v in language_inputs.items()})
        inputs["proprio"] = self.torch.as_tensor(state, dtype=self.torch.float32).unsqueeze(0).to(self.device)
        return inputs

    def get_action(self, obs: Dict[str, object]) -> np.ndarray:
        """
        Predict a future qpos chunk from the current observation.
        """
        inputs = self._prepare_inputs(obs)
        with self.torch.no_grad():
            action_chunk = self.model.generate_actions(
                input_ids=inputs["input_ids"],
                image_input=inputs["image_input"],
                image_mask=inputs["image_mask"],
                proprio=inputs["proprio"],
                steps=self.integration_steps,
            )

        action_chunk = action_chunk.squeeze(0).float().cpu().numpy()
        execute_horizon = min(self.execute_horizon, len(action_chunk))
        return np.asarray(action_chunk[:execute_horizon], dtype=np.float32)


def get_model(usr_args):
    """
    Load a local SimVLA RMBench policy for server-side inference.
    """
    checkpoint_path = usr_args["checkpoint_path"]
    smolvlm_model_path = usr_args.get("smolvlm_model_path")
    norm_stats_path = usr_args.get("norm_stats_path")
    device = usr_args.get("device", "cuda")
    execute_horizon = usr_args.get("execute_horizon", 5)
    integration_steps = usr_args.get("integration_steps", 10)
    camera_keys = usr_args.get("camera_keys", list(DEFAULT_CAMERA_KEYS))
    expected_state_dim = usr_args.get("expected_state_dim", 14)

    return SimVLAPolicyRunner(
        checkpoint_path=checkpoint_path,
        smolvlm_model_path=smolvlm_model_path,
        norm_stats_path=norm_stats_path,
        device=device,
        execute_horizon=execute_horizon,
        integration_steps=integration_steps,
        camera_keys=camera_keys,
        expected_state_dim=expected_state_dim,
    )


__all__ = ["DEFAULT_CAMERA_KEYS", "SimVLAPolicyRunner", "get_model"]
