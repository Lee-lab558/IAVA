#!/bin/bash


## set below
####################################################
seed=42
dataset_name="coco"  # coco | aokvqa | gqa
type="adversarial" # random | popular | adversarial
model="instructblip" # llava | qwen-vl | instructblip
use_IAVA=True
layer_gamma=0.5
masking_scheme="zeros"			# "10:1" or "-1"
lamb=2.0
gpus=5
max_token=64
log_path="/pope-instructblip-test/test2/${type}"
cd_alpha=1.0
cd_beta=0.1
cd_alpha_list=(3.0)
conv=llava_v1
batch_size=1
masking_method="zeros"
use_cd=False
exp_description=".."


model_path="/model_checkpoints/instructblip-vicuna-7b"
pope_path="/base_models/LLaVA/playground/data/eval/pope/${dataset_name}/${dataset_name}_pope_${type}.json"
data_path="/base_models/LLaVA/playground/data/eval/pope/val2014"
####################################################



export CUDA_VISIBLE_DEVICES=${gpus}
python ./eval_bench/pope_eval_${model}b.py \
--seed ${seed} \
--model-path ${model_path} \
--pope_path ${pope_path} \
--data_path ${data_path} \
--log_path ${log_path} \
--conv-mode ${conv} \
--batch-size ${batch_size} \
--use_IAVA ${use_IAVA} \
--layer_gamma ${layer_gamma} \
--masking_scheme ${masking_method} \
--lamb ${lamb} \
--use_cd ${use_cd} \
--exp_description ${exp_description} \
--max_token ${max_token} \
--cd_alpha ${cd_alpha} \
--cd_beta ${cd_beta} \