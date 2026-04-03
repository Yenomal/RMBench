"""
Client-side adapter for running OpenPI inside RMBench.

Responsibilities:
- translate RMBench observations into OpenPI Aloha-style inputs
- call either a local runner or a remote ModelClient
- execute qpos action chunks in the environment
"""

from __future__ import annotations

from typing import Any
from typing import Dict

import numpy as np


def get_model(usr_args):
    from .server_adapter import get_model as server_get_model

    return server_get_model(usr_args)


def _to_chw(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape={image.shape}")
    if image.shape[0] == 3:
        return image
    return np.transpose(image, (2, 0, 1))


def encode_obs(observation: dict, instruction: str | None = None) -> Dict[str, object]:
    obs = {
        "images": {
            "cam_high": _to_chw(observation["observation"]["head_camera"]["rgb"]),
            "cam_left_wrist": _to_chw(observation["observation"]["left_camera"]["rgb"]),
            "cam_right_wrist": _to_chw(observation["observation"]["right_camera"]["rgb"]),
        },
        "state": np.asarray(observation["joint_action"]["vector"], dtype=np.float32),
    }
    if instruction:
        obs["prompt"] = instruction
    return obs


def _request_action(model: Any, encoded_obs: Dict[str, object]) -> np.ndarray:
    if hasattr(model, "call"):
        actions = model.call(func_name="get_action", obs=encoded_obs)
    else:
        actions = model.get_action(encoded_obs)
    return np.asarray(actions, dtype=np.float32)


def eval(TASK_ENV, model, observation):
    instruction = TASK_ENV.get_instruction()
    encoded_obs = encode_obs(observation, instruction=instruction)
    action_chunk = _request_action(model, encoded_obs)

    for action in action_chunk:
        TASK_ENV.take_action(action, action_type="qpos")
        if TASK_ENV.eval_success:
            break


def reset_model(model):
    if hasattr(model, "call"):
        return model.call(func_name="reset_model")
    return model.reset_model()


__all__ = ["get_model", "encode_obs", "eval", "reset_model"]
