#!/bin/bash

policy_name=SimVLA_AutoGaze
task_name=${1}
task_config=${2}
ckpt_setting=${3}
seed=${4}
gpu_id=${5}
checkpoint_path=${6}
smolvlm_model_path=${7}
norm_stats_path=${8}
instruction_type=${9:-unseen}
execute_horizon=${10:-5}
integration_steps=${11:-10}
autogaze_model_path=${12:-nvidia/AutoGaze}
autogaze_siglip_model_path=${13:-google/siglip2-base-patch16-224}
autogaze_history_len=${14:-8}
autogaze_projector_hidden_size=${15:-1536}
autogaze_gazing_ratio=${16:-0.1}

export CUDA_VISIBLE_DEVICES=${gpu_id}

cd ../..

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/${policy_name}/deploy_policy.yml \
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

