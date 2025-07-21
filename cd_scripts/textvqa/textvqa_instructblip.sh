#!/bin/bash
test=vcd
gpus=6
export CUDA_VISIBLE_DEVICES=${gpus}

python /experiments/eval/model_vqa_loader_instructblip.py \
    --model-path /model_checkpoints/instructblip-vicuna-7b \
    --question-file /base_models/LLaVA/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder /base_models/LLaVA/playground/data/eval/textvqa/train_images \
    --answers-file /textvqa_answers_instructblip/${test}/${test}.jsonl \
    --log_path /textvqa_answers_instructblip/${test} \
    --num_beams 1 \
    --temperature 1.0 \
    --conv-mode llava_v1

python /experiments/eval/eval_textvqa.py \
    --annotation-file /base_models/LLaVA/playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file /textvqa_answers_instructblip/${test}/${test}.jsonl
