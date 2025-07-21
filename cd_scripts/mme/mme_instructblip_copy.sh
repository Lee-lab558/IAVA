#!/bin/bash

seed=42
dataset_name="mme"
question_file="/base_models/LLaVA/playground/data/eval/MME/llava_mme.jsonl"
image_folder="/base_models/LLaVA/playground/data/eval/MME/MME_Benchmark_release_version"

## max token maybe important
### set below ###
model="instructblip" # llava | qwenvl | instructblip
use_IAVA=True
layer_gamma=0.5
masking_scheme="zeros"
max_token=64
use_cd=False
cd_alpha=1.0
gpus=6
use_m3id=False
model_path="/model_checkpoints/instructblip-vicuna-7b"
log_path="/mme_instructblip_test/"
answer_path="${log_path}"

#################

export CUDA_VISIBLE_DEVICES=${gpus}

# Loop for lambda values from 0 to 31
for lamb in {0..31}
do
    echo "Running with lambda = $lamb"
    
    # Run the evaluation script with the current lambda value
    python ./experiments/eval/mme_${model}.py \
    --seed ${seed} \
    --model-path ${model_path} \
    --question-file ${question_file} \
    --image-folder ${image_folder} \
    --answers-file ${answer_path}/${lamb}/answer.jsonl \
    --use_IAVA ${use_IAVA} \
    --layer_gamma ${layer_gamma} \
    --masking_scheme ${masking_scheme} \
    --lamb ${lamb} \
    --cd_alpha ${cd_alpha} \
    --max_token ${max_token} \
    --use_cd ${use_cd} \
    --use_m3id ${use_m3id} \
    --log_path ${log_path}/${lamb} 

    # Convert answers to the MME format
    python ./experiments/eval/convert_answer_to_mme.py \
    --output_path ${answer_path}/${lamb}/answer.jsonl \
    --seed ${seed} \
    --log_path ${answer_path}/${lamb}

    # Run the calculation script
    python ./experiments/eval/calculation.py \
    --results_dir ${answer_path}/${lamb}/mme_answers
done
