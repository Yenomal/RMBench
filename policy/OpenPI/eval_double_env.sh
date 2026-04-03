#!/bin/bash
set -euo pipefail

policy_name=OpenPI
task_name=${1}
task_config=${2}
ckpt_setting=${3}
expert_data_num=${4}
seed=${5}
gpu_id=${6}
openpi_python=${7:-/home/rui/Yenomal/third_party/openpi/.venv/bin/python}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../..

FREE_PORT=$(python3 - << 'EOF'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('', 0))
    print(s.getsockname()[1])
EOF
)
echo -e "\033[33mUsing socket port: ${FREE_PORT}\033[0m"

echo -e "\033[32m[server] Launching policy_model_server with ${openpi_python}\033[0m"
PYTHONWARNINGS=ignore::UserWarning \
${openpi_python} script/policy_model_server.py \
    --port ${FREE_PORT} \
    --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name} &
SERVER_PID=$!

trap "echo -e '\033[31m[cleanup] Killing server (PID=${SERVER_PID})\033[0m'; kill ${SERVER_PID} 2>/dev/null || true" EXIT

echo -e "\033[34m[client] Starting eval_policy_client on port ${FREE_PORT}...\033[0m"
PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy_client.py \
    --port ${FREE_PORT} \
    --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name}

echo -e "\033[33m[main] eval_policy_client has finished; the server will be terminated.\033[0m"
