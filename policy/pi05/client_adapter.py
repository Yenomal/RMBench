"""
Client-side adapter for running pi05 inside RMBench.

Responsibilities:
- translate RMBench observations into transport-friendly payloads
- call either a local model or a remote ModelClient
- execute action chunks inside TASK_ENV
"""

from __future__ import annotations

from typing import Dict

import numpy as np


DEFAULT_CAMERA_KEYS = ("head_camera", "left_camera", "right_camera")


def get_model(usr_args):
    """Local-mode compatibility entry."""
    from .server_adapter import get_model as server_get_model

    return server_get_model(usr_args)


def encode_obs(observation: dict, instruction: str | None = None) -> Dict[str, object]:
    obs_block = observation.get("observation", {})
    encoded: Dict[str, object] = {}

    for camera_key in DEFAULT_CAMERA_KEYS:
        camera = obs_block.get(camera_key, {})
        rgb = camera.get("rgb")
        if rgb is not None:
            encoded[camera_key] = np.asarray(rgb, dtype=np.uint8)

    if not any(key in encoded for key in DEFAULT_CAMERA_KEYS):
        raise KeyError("No RGB camera found in observation")

    joint_vec = observation.get("joint_action", {}).get("vector")
    if joint_vec is None:
        raise KeyError("joint_action.vector missing from observation")

    encoded["state"] = np.asarray(joint_vec, dtype=np.float32)
    encoded["instruction"] = "" if instruction is None else str(instruction)
    return encoded


def _request_action(model, encoded_obs: Dict[str, object]) -> np.ndarray:
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


__all__ = ["DEFAULT_CAMERA_KEYS", "get_model", "encode_obs", "eval", "reset_model"]
