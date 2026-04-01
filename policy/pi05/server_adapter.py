"""
Server-side adapter for serving pi05 to RMBench.

Responsibilities:
- load the project-local pi05 runtime
- maintain head-camera history cache inside the server runtime
- convert requests into action chunks
"""

from __future__ import annotations

DEFAULT_CAMERA_KEYS = ("head_camera", "left_camera", "right_camera")


class Pi05PolicyRunner:
    def __init__(
        self,
        train_config_name: str,
        model_name: str,
        checkpoint_id: str | int,
        pi0_step: int = 50,
        checkpoint_root: str | None = None,
        device: str | None = None,
    ):
        from .pi_model import PI0

        self.runner = PI0(
            train_config_name=train_config_name,
            model_name=model_name,
            checkpoint_id=checkpoint_id,
            pi0_step=pi0_step,
            checkpoint_root=checkpoint_root,
            pytorch_device=device,
        )

    def get_action(self, obs: dict):
        return self.runner.get_action(obs)

    def reset_model(self):
        return self.runner.reset_model()


def get_model(usr_args):
    checkpoint_root = usr_args.get("checkpoint_root")
    device = usr_args.get("device", "cuda")

    return Pi05PolicyRunner(
        train_config_name=usr_args["train_config_name"],
        model_name=usr_args["model_name"],
        checkpoint_id=usr_args.get("checkpoint_id", "latest"),
        pi0_step=usr_args.get("pi0_step", 50),
        checkpoint_root=checkpoint_root,
        device=device,
    )


__all__ = ["DEFAULT_CAMERA_KEYS", "Pi05PolicyRunner", "get_model"]
