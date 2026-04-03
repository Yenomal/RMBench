"""
Server-side adapter for serving AutoGaze+SimVLA to RMBench.

Responsibilities:
- load SimVLA + AutoGaze observation encoder configuration
- maintain head-camera history cache
- convert RMBench observations into model inputs
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
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "SimVLA"))


def _load_checkpoint_config(checkpoint_path: str):
    config_path = os.path.join(checkpoint_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config.json under checkpoint path: {checkpoint_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_checkpoint_state_dict(checkpoint_path: str, torch_module):
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


class SimVLAAutoGazePolicyRunner:
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
        autogaze_model_path: str = "nvidia/AutoGaze",
        autogaze_siglip_model_path: str = "google/siglip2-base-patch16-224",
        autogaze_history_len: int = 8,
        autogaze_projector_hidden_size: int = 1536,
        autogaze_gazing_ratio: float = 0.1,
        autogaze_task_loss_requirement: float | None = None,
    ):
        simvla_root = _simvla_root()
        if simvla_root not in sys.path:
            sys.path.insert(0, simvla_root)

        import torch
        import torch.nn.functional as F
        from models.configuration_smolvlm_vla import SmolVLMVLAConfig
        from models.modeling_smolvlm_vla import SmolVLMVLA
        from models.processing_smolvlm_vla import SmolVLMVLAProcessor

        self.torch = torch
        self.F = F
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.camera_keys = list(camera_keys)
        self.execute_horizon = max(1, int(execute_horizon))
        self.integration_steps = max(1, int(integration_steps))
        self.expected_state_dim = int(expected_state_dim)
        self.history_len = int(autogaze_history_len)
        self.head_history = deque(maxlen=self.history_len)

        config_dict = _load_checkpoint_config(checkpoint_path)
        if smolvlm_model_path:
            config_dict["smolvlm_model_path"] = smolvlm_model_path

        config_dict.update(
            {
                "use_autogaze_obs_encoder": True,
                "autogaze_model_path": autogaze_model_path,
                "autogaze_siglip_model_path": autogaze_siglip_model_path,
                "autogaze_history_len": self.history_len,
                "autogaze_projector_hidden_size": int(autogaze_projector_hidden_size),
                "autogaze_gazing_ratio": float(autogaze_gazing_ratio),
                "autogaze_task_loss_requirement": autogaze_task_loss_requirement,
            }
        )

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
        self.image_size = int(getattr(self.model.config, "image_size", 384))
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self.device).view(1, 3, 1, 1)

        if norm_stats_path:
            self.model.action_space.load_norm_stats(norm_stats_path)

    def reset_model(self):
        self.head_history.clear()
        return True

    def _preprocess_frame(self, image: np.ndarray):
        tensor = self.torch.as_tensor(image, device=self.device, dtype=self.torch.float32)
        if tensor.ndim != 3:
            raise ValueError(f"Expected image ndim=3, got shape={tuple(tensor.shape)}")
        tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
        tensor = self.F.interpolate(
            tensor,
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        tensor = (tensor - self.mean) / self.std
        return tensor.squeeze(0)

    def _build_history_tensor(self, current_head: np.ndarray):
        if len(self.head_history) == 0:
            frames = [current_head for _ in range(self.history_len)]
        else:
            frames = list(self.head_history)
            if len(frames) < self.history_len:
                frames = [frames[0]] * (self.history_len - len(frames)) + frames
            else:
                frames = frames[-self.history_len :]

        hist = [self._preprocess_frame(frame) for frame in frames]
        return self.torch.stack(hist, dim=0).unsqueeze(0)

    def _prepare_inputs(self, obs: Dict[str, object]) -> Dict[str, object]:
        images = [obs[k] for k in self.camera_keys if k in obs]
        if len(images) == 0:
            raise ValueError("At least one camera image is required")

        state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
        if state.shape[0] != self.expected_state_dim:
            raise ValueError(
                f"Expected state dim {self.expected_state_dim}, got {state.shape[0]}. "
                "This baseline currently targets the default aloha-agilex embodiment."
            )

        processed_images = [self._preprocess_frame(img) for img in images]
        while len(processed_images) < len(self.camera_keys):
            processed_images.append(self.torch.zeros_like(processed_images[0]))
        image_input = self.torch.stack(processed_images[: len(self.camera_keys)], dim=0).unsqueeze(0)
        image_mask = self.torch.zeros(1, len(self.camera_keys), dtype=self.torch.bool, device=self.device)
        image_mask[:, : len(images)] = True

        current_head = np.asarray(obs["head_camera"], dtype=np.uint8)
        head_history = self._build_history_tensor(current_head)

        language_inputs = self.processor.encode_language(obs.get("instruction", ""))
        input_ids = self.torch.as_tensor(language_inputs["input_ids"], device=self.device)
        proprio = self.torch.as_tensor(state, dtype=self.torch.float32, device=self.device).unsqueeze(0)

        return {
            "input_ids": input_ids,
            "image_input": image_input,
            "image_mask": image_mask,
            "head_history": head_history,
            "proprio": proprio,
        }

    def get_action(self, obs: Dict[str, object]) -> np.ndarray:
        inputs = self._prepare_inputs(obs)
        with self.torch.no_grad():
            action_chunk = self.model.generate_actions(
                input_ids=inputs["input_ids"],
                image_input=inputs["image_input"],
                image_mask=inputs["image_mask"],
                head_history=inputs["head_history"],
                proprio=inputs["proprio"],
                steps=self.integration_steps,
            )

        self.head_history.append(np.asarray(obs["head_camera"], dtype=np.uint8))

        action_chunk = action_chunk.squeeze(0).float().cpu().numpy()
        execute_horizon = min(self.execute_horizon, len(action_chunk))
        return np.asarray(action_chunk[:execute_horizon], dtype=np.float32)


def get_model(usr_args):
    checkpoint_path = usr_args["checkpoint_path"]
    smolvlm_model_path = usr_args.get("smolvlm_model_path")
    norm_stats_path = usr_args.get("norm_stats_path")
    device = usr_args.get("device", "cuda")
    execute_horizon = usr_args.get("execute_horizon", 5)
    integration_steps = usr_args.get("integration_steps", 10)
    camera_keys = usr_args.get("camera_keys", list(DEFAULT_CAMERA_KEYS))
    expected_state_dim = usr_args.get("expected_state_dim", 14)

    return SimVLAAutoGazePolicyRunner(
        checkpoint_path=checkpoint_path,
        smolvlm_model_path=smolvlm_model_path,
        norm_stats_path=norm_stats_path,
        device=device,
        execute_horizon=execute_horizon,
        integration_steps=integration_steps,
        camera_keys=camera_keys,
        expected_state_dim=expected_state_dim,
        autogaze_model_path=usr_args.get("autogaze_model_path", "nvidia/AutoGaze"),
        autogaze_siglip_model_path=usr_args.get("autogaze_siglip_model_path", "google/siglip2-base-patch16-224"),
        autogaze_history_len=usr_args.get("autogaze_history_len", 8),
        autogaze_projector_hidden_size=usr_args.get("autogaze_projector_hidden_size", 1536),
        autogaze_gazing_ratio=usr_args.get("autogaze_gazing_ratio", 0.1),
        autogaze_task_loss_requirement=usr_args.get("autogaze_task_loss_requirement", None),
    )


__all__ = ["DEFAULT_CAMERA_KEYS", "SimVLAAutoGazePolicyRunner", "get_model"]

