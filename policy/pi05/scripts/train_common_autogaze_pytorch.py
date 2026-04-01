import argparse
import dataclasses
import json
import logging
import math
from pathlib import Path
import shutil
from typing import Iterable

import torch
from safetensors.torch import save_file
import wandb

import openpi.models.model as _model
from openpi.models_pytorch import pi0_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


LOGGER = logging.getLogger("pi05_common_autogaze_pytorch")
PROJECTOR_PREFIX = "common_visual_prefix.stack.projector.net."


def init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def warmup_cosine_lr(step: int, *, warmup_steps: int, total_steps: int, peak_lr: float, final_lr: float) -> float:
    if total_steps <= 1:
        return peak_lr
    if step < warmup_steps:
        return peak_lr * float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return final_lr + (peak_lr - final_lr) * cosine


def build_config(args: argparse.Namespace) -> _config.TrainConfig:
    base_config = _config.get_config(args.train_config_name)
    model_cfg = dataclasses.replace(
        base_config.model,
        use_common_visual_encoder=True,
        common_visual_history_len=args.history_len,
    )
    data_cfg = dataclasses.replace(
        base_config.data,
        repo_id=args.repo_id,
        assets=dataclasses.replace(base_config.data.assets, asset_id=args.asset_id or args.repo_id),
    )
    return dataclasses.replace(
        base_config,
        exp_name=args.model_name,
        model=model_cfg,
        data=data_cfg,
        assets_base_dir=args.assets_base_dir,
        checkpoint_base_dir=args.checkpoint_base_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_steps=args.num_steps,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        wandb_enabled=not args.disable_wandb,
        pytorch_training_precision=args.precision,
    )


def load_model(config: _config.TrainConfig, init_weight_path: str | None, device: torch.device) -> torch.nn.Module:
    if init_weight_path:
        model = config.model.load_pytorch(config, init_weight_path)
    else:
        model = pi0_pytorch.PI0Pytorch(config.model)
    return model.to(device)


def set_trainable_parameters(model: torch.nn.Module, stage: str) -> dict[str, list[torch.nn.Parameter]]:
    for param in model.parameters():
        param.requires_grad = False

    groups: dict[str, list[torch.nn.Parameter]] = {
        "visual": [],
        "action": [],
        "siglip": [],
    }

    siglip_layer_prefixes = [f"common_visual_prefix.stack.siglip.vision_model.encoder.layers.{idx}." for idx in range(8, 12)]
    extra_siglip_prefixes = (
        "common_visual_prefix.stack.siglip.vision_model.post_layernorm.",
    )

    for name, param in model.named_parameters():
        if name.startswith("common_visual_prefix.stack.projector.") or name.startswith(
            "common_visual_prefix.stack.source_embed"
        ) or name.startswith("common_visual_prefix.stack.age_embed"):
            param.requires_grad = True
            groups["visual"].append(param)
            continue

        if stage in {"stage_b", "stage_c"} and (
            name.startswith("paligemma_with_expert.gemma_expert.")
            or name.startswith("action_in_proj.")
            or name.startswith("action_out_proj.")
            or name.startswith("time_mlp_in.")
            or name.startswith("time_mlp_out.")
            or name.startswith("action_time_mlp_in.")
            or name.startswith("action_time_mlp_out.")
            or name.startswith("state_proj.")
        ):
            param.requires_grad = True
            groups["action"].append(param)
            continue

        if stage == "stage_c" and (
            any(name.startswith(prefix) for prefix in siglip_layer_prefixes)
            or any(name.startswith(prefix) for prefix in extra_siglip_prefixes)
        ):
            param.requires_grad = True
            groups["siglip"].append(param)

    return groups


def build_optimizer(
    groups: dict[str, list[torch.nn.Parameter]],
    *,
    stage: str,
    visual_lr: float,
    action_lr: float,
    siglip_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    param_groups = []
    if groups["visual"]:
        param_groups.append({"params": groups["visual"], "lr": visual_lr, "name": "visual"})
    if stage in {"stage_b", "stage_c"} and groups["action"]:
        param_groups.append({"params": groups["action"], "lr": action_lr, "name": "action"})
    if stage == "stage_c" and groups["siglip"]:
        param_groups.append({"params": groups["siglip"], "lr": siglip_lr, "name": "siglip"})
    if not param_groups:
        raise ValueError(f"No trainable parameters found for stage {stage}")
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def observation_to_device(observation: _model.Observation, device: torch.device) -> _model.Observation:
    return _model.Observation(
        images={k: v.to(device) for k, v in observation.images.items()},
        image_masks={k: v.to(device) for k, v in observation.image_masks.items()},
        state=observation.state.to(device),
        head_history=None if observation.head_history is None else observation.head_history.to(device),
        tokenized_prompt=None if observation.tokenized_prompt is None else observation.tokenized_prompt.to(device),
        tokenized_prompt_mask=None
        if observation.tokenized_prompt_mask is None
        else observation.tokenized_prompt_mask.to(device),
        token_ar_mask=None if observation.token_ar_mask is None else observation.token_ar_mask.to(device),
        token_loss_mask=None if observation.token_loss_mask is None else observation.token_loss_mask.to(device),
    )


def compute_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    found = False
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach().float()
        total += grad.pow(2).sum().item()
        found = True
    if not found:
        return 0.0
    return total ** 0.5


def projector_grad_norm(model: torch.nn.Module) -> float:
    params = [param for name, param in model.named_parameters() if name.startswith(PROJECTOR_PREFIX)]
    return compute_grad_norm(params)


def save_checkpoint(
    model: torch.nn.Module,
    checkpoint_dir: Path,
    step: int,
    *,
    data_config: _config.DataConfig,
    train_config: _config.TrainConfig,
    latest_only: bool = False,
) -> None:
    target_dir = checkpoint_dir / ("latest" if latest_only else str(step))
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    state_dict = {name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()}
    save_file(state_dict, str(target_dir / "model.safetensors"))

    assets_dir = target_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    if data_config.norm_stats is not None and data_config.asset_id is not None:
        _normalize.save(assets_dir / data_config.asset_id, data_config.norm_stats)

    metadata = {
        "step": step,
        "train_config_name": train_config.name,
        "model_name": train_config.exp_name,
        "asset_id": data_config.asset_id,
        "use_common_visual_encoder": True,
        "common_visual_history_len": train_config.model.common_visual_history_len,
    }
    (target_dir / "train_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config-name", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--asset-id", default=None)
    parser.add_argument("--stage", choices=("stage_a", "stage_b", "stage_c"), default="stage_a")
    parser.add_argument("--history-len", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--visual-lr", type=float, default=1e-4)
    parser.add_argument("--action-lr", type=float, default=2e-5)
    parser.add_argument("--siglip-lr", type=float, default=5e-6)
    parser.add_argument("--final-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--init-weight-path", default=None)
    parser.add_argument("--checkpoint-base-dir", default="./policy/pi05/checkpoints")
    parser.add_argument("--assets-base-dir", default="./policy/pi05/assets")
    parser.add_argument("--wandb-project", default="pi05-common-autogaze")
    parser.add_argument("--disable-wandb", action="store_true")
    args = parser.parse_args()

    init_logging()
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    config = build_config(args)
    config = dataclasses.replace(config, project_name=args.wandb_project)

    data_loader = _data_loader.create_data_loader(config, shuffle=True, framework="pytorch")
    data_iter = iter(data_loader)
    data_config = data_loader.data_config()

    model = load_model(config, args.init_weight_path, device)
    trainable_groups = set_trainable_parameters(model, args.stage)
    optimizer = build_optimizer(
        trainable_groups,
        stage=args.stage,
        visual_lr=args.visual_lr,
        action_lr=args.action_lr,
        siglip_lr=args.siglip_lr,
        weight_decay=args.weight_decay,
    )

    if not args.disable_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.model_name,
            config={
                "train_config_name": args.train_config_name,
                "repo_id": args.repo_id,
                "stage": args.stage,
                "history_len": args.history_len,
                "batch_size": args.batch_size,
                "num_steps": args.num_steps,
                "visual_lr": args.visual_lr,
                "action_lr": args.action_lr,
                "siglip_lr": args.siglip_lr,
            },
        )

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(f"Training on device={device}, stage={args.stage}, checkpoint_dir={checkpoint_dir}")
    LOGGER.info(
        "Trainable parameter groups: visual=%d action=%d siglip=%d",
        len(trainable_groups["visual"]),
        len(trainable_groups["action"]),
        len(trainable_groups["siglip"]),
    )

    model.train()
    for step in range(1, args.num_steps + 1):
        observation, actions = next(data_iter)
        observation = observation_to_device(observation, device)
        actions = actions.to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = model(observation, actions).mean()
        loss.backward()

        grad_norm_value = projector_grad_norm(model)
        torch.nn.utils.clip_grad_norm_(
            [param for param in model.parameters() if param.requires_grad],
            max_norm=args.grad_clip_norm,
        )
        optimizer.step()

        for param_group in optimizer.param_groups:
            if param_group["name"] == "visual":
                peak_lr = args.visual_lr
            elif param_group["name"] == "action":
                peak_lr = args.action_lr
            else:
                peak_lr = args.siglip_lr
            param_group["lr"] = warmup_cosine_lr(
                step,
                warmup_steps=args.warmup_steps,
                total_steps=args.num_steps,
                peak_lr=peak_lr,
                final_lr=peak_lr * args.final_lr_ratio,
            )

        if step % args.log_interval == 0 or step == 1:
            log_payload = {
                "loss": float(loss.item()),
                "grad_norm": float(grad_norm_value),
                "lr_visual": float(next(group["lr"] for group in optimizer.param_groups if group["name"] == "visual")),
                "step": step,
            }
            action_group = next((group for group in optimizer.param_groups if group["name"] == "action"), None)
            siglip_group = next((group for group in optimizer.param_groups if group["name"] == "siglip"), None)
            if action_group is not None:
                log_payload["lr_action"] = float(action_group["lr"])
            if siglip_group is not None:
                log_payload["lr_siglip"] = float(siglip_group["lr"])
            LOGGER.info(
                "step=%d loss=%.6f projector_grad_norm=%.6f",
                step,
                log_payload["loss"],
                log_payload["grad_norm"],
            )
            if not args.disable_wandb:
                wandb.log(log_payload, step=step)

        if step % args.save_interval == 0 or step == args.num_steps:
            save_checkpoint(model, checkpoint_dir, step, data_config=data_config, train_config=config, latest_only=False)
            save_checkpoint(model, checkpoint_dir, step, data_config=data_config, train_config=config, latest_only=True)

    if not args.disable_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
