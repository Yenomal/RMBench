#!/bin/bash

policy_name=SimVLA
task_name=${1}
task_config=${2}
ckpt_setting=${3}
seed=${4}
server_env=${5}
client_env=${6}
checkpoint_path=${7}
smolvlm_model_path=${8}
norm_stats_path=${9}
instruction_type=${10:-unseen}
execute_horizon=${11:-5}
integration_steps=${12:-10}

cd ../..

FREE_PORT=$(python3 - << 'EOF'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('', 0))
    print(s.getsockname()[1])
EOF
)

echo -e "\033[33mUsing socket port: ${FREE_PORT}\033[0m"
echo -e "\033[32mStarting SimVLA server in env: ${server_env}\033[0m"

conda run -n "${server_env}" python script/policy_model_server.py \
    --port ${FREE_PORT} \
    --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --instruction_type ${instruction_type} \
    --checkpoint_path ${checkpoint_path} \
    --smolvlm_model_path ${smolvlm_model_path} \
    --norm_stats_path ${norm_stats_path} \
    --execute_horizon ${execute_horizon} \
    --integration_steps ${integration_steps} &
SERVER_PID=$!

trap "echo -e '\033[31m[cleanup] Killing server (PID=${SERVER_PID})\033[0m'; kill ${SERVER_PID} 2>/dev/null" EXIT

echo -e "\033[34mStarting RMBench client in env: ${client_env}\033[0m"
conda run -n "${client_env}" python script/eval_policy_client.py \
    --port ${FREE_PORT} \
    --config policy/${policy_name}/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --seed ${seed} \
    --policy_name ${policy_name} \
    --instruction_type ${instruction_type} \
    --checkpoint_path ${checkpoint_path} \
    --smolvlm_model_path ${smolvlm_model_path} \
    --norm_stats_path ${norm_stats_path} \
    --execute_horizon ${execute_horizon} \
    --integration_steps ${integration_steps}
