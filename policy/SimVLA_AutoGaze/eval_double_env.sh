#!/bin/bash

policy_name=SimVLA_AutoGaze
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
autogaze_model_path=${13:-nvidia/AutoGaze}
autogaze_siglip_model_path=${14:-google/siglip2-base-patch16-224}
autogaze_history_len=${15:-8}
autogaze_projector_hidden_size=${16:-1536}
autogaze_gazing_ratio=${17:-0.1}

cd ../..

FREE_PORT=$(python3 - << 'EOF'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(('', 0))
    print(s.getsockname()[1])
EOF
)

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
    --integration_steps ${integration_steps} \
    --autogaze_model_path ${autogaze_model_path} \
    --autogaze_siglip_model_path ${autogaze_siglip_model_path} \
    --autogaze_history_len ${autogaze_history_len} \
    --autogaze_projector_hidden_size ${autogaze_projector_hidden_size} \
    --autogaze_gazing_ratio ${autogaze_gazing_ratio} &
SERVER_PID=$!

trap "kill ${SERVER_PID} 2>/dev/null" EXIT

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
    --integration_steps ${integration_steps} \
    --autogaze_model_path ${autogaze_model_path} \
    --autogaze_siglip_model_path ${autogaze_siglip_model_path} \
    --autogaze_history_len ${autogaze_history_len} \
    --autogaze_projector_hidden_size ${autogaze_projector_hidden_size} \
    --autogaze_gazing_ratio ${autogaze_gazing_ratio}
