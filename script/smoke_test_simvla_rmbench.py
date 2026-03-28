#!/usr/bin/env python
"""
Minimal end-to-end smoke test for the SimVLA <-> RMBench integration.

This script intentionally does not run a full benchmark sweep. It only checks
that the following loop works for a few replans:

  RMBench env -> observation -> SimVLA server -> action chunk -> env.step
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml

sys.path.append("./")

from envs import CONFIGS_PATH
from script.eval_policy import class_decorator, get_embodiment_config
from script.eval_policy_client import ModelClient


def build_args(task_name: str, task_config: str) -> dict:
    """Load RMBench task config and expand embodiment paths."""
    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.safe_load(f)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = "smoke"
    args["render_freq"] = 0
    args["eval_mode"] = True
    args["eval_video_log"] = False
    args["clear_cache_freq"] = 1

    embodiment_type = args.get("embodiment")
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        embodiment_types = yaml.safe_load(f)

    def get_embodiment_file(emb_type):
        return embodiment_types[emb_type]["file_path"]

    if len(embodiment_type) != 1:
        raise RuntimeError("This smoke test only supports the default single embodiment entry")

    args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
    args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
    args["dual_arm_embodied"] = True
    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    return args


def run_smoke_test(
    task_name: str,
    task_config: str,
    port: int,
    seed: int,
    instruction: str,
    max_replans: int,
) -> int:
    """Run a few SimVLA replans against an RMBench environment."""
    args = build_args(task_name, task_config)
    task = class_decorator(task_name)
    task.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
    task.set_instruction(instruction)

    client = ModelClient(host="localhost", port=port, timeout=120)

    try:
        for step in range(max_replans):
            obs = task.get_obs()
            payload = {
                "head_camera": obs["observation"]["head_camera"]["rgb"],
                "left_camera": obs["observation"]["left_camera"]["rgb"],
                "right_camera": obs["observation"]["right_camera"]["rgb"],
                "state": obs["joint_action"]["vector"],
                "instruction": instruction,
            }

            actions = client.call(func_name="get_action", obs=payload)
            actions = np.asarray(actions, dtype=np.float32)
            print(
                f"[smoke] replan={step} chunk_shape={tuple(actions.shape)} "
                f"first_action_shape={tuple(actions[0].shape)}"
            )

            for action in actions:
                task.take_action(action, action_type="qpos")
                if task.eval_success:
                    print("[smoke] task reached eval_success=True")
                    return 0

        print(f"[smoke] finished {max_replans} replans, eval_success={task.eval_success}, "
              f"take_action_cnt={task.take_action_cnt}")
        return 0
    finally:
        client.close()
        task.close_env(clear_cache=False)


def main():
    parser = argparse.ArgumentParser(description="SimVLA RMBench smoke test")
    parser.add_argument("--task_name", type=str, default="battery_try")
    parser.add_argument("--task_config", type=str, default="demo_clean")
    parser.add_argument("--port", type=int, default=19000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_replans", type=int, default=3)
    parser.add_argument(
        "--instruction",
        type=str,
        default="There are two batteries and a battery slot on the table. Combining the two batteries in different orientations causes the dashboard needle to rotate.",
    )
    args = parser.parse_args()

    raise SystemExit(
        run_smoke_test(
            task_name=args.task_name,
            task_config=args.task_config,
            port=args.port,
            seed=args.seed,
            instruction=args.instruction,
            max_replans=args.max_replans,
        )
    )


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
