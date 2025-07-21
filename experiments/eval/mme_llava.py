import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math
import torch.distributed as dist
from utils import dist_util
from utils.logger import create_logger
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

# import kornia
from transformers import set_seed
from IAVA_utils.vcd_add_noise import add_diffusion_noise
from IAVA_utils.IAVA_sample import evolve_IAVA_sampling
evolve_IAVA_sampling()

def recorder(out):
    NEG_WORDS = ["No", "not", "no", "NO"]

    out = out.replace('.', '')
    out = out.replace(',', '')
    words = out.split(' ')
    if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
        return "No"
    else:
        return "Yes"


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def eval_model(args):
    
    # set up gpu and logging
    
    dist_util.setup_dist(args)
    device = dist_util.device()

    # Setup an experiment folder:
    if dist.get_rank() == 0:
        os.makedirs(
            args.log_path, exist_ok=True
        )  # Make results folder (holds all experiment subfolders)
        model_string_name = args.model_path.split("/")[-1]
        experiment_index = len(glob(f"{args.log_path}/{model_string_name}/*"))
        experiment_dir = f"{args.log_path}"  # Create an experiment folder
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)
    logger.info(f"use_cd: {args.use_cd}, method: {args.use_IAVA}, layer_gamma: {args.layer_gamma}, masking_scheme: {args.masking_scheme}, lamb: {args.lamb}")
    logger.info(f"question_file : {args.question_file}")


    
    
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    model.image_processor = image_processor

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    # questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    for line in tqdm(questions):
    # for (input_ids, image_tensor, image_sizes), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        cur_prompt = qs
        
        
        # one word processing
        qs = qs.split('\n')[0] 
        cur_prompt = qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        # conv.append_message(conv.roles[0], qs + " Please answer this question with one word.")
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        # 创建一个不含问题文本的prompt1，进而生成input_ids1
        # 先分割qs，保留 '<image>' 部分
        qs_parts = qs.split('\n')
        image_part = qs_parts[0]

        # 构建新的 qs1，在末尾追加描述请求
        qs1 = f"{image_part}\nPlease describe the content of the picture."

        # 根据模型配置决定如何构建qs
        if model.config.mm_use_im_start_end:
            qs1 = f"{DEFAULT_IM_START_TOKEN}{qs1}{DEFAULT_IM_END_TOKEN}"
        else:
            qs1 = qs1

        # 准备对话模板
        conv = conv_templates[args.conv_mode].copy()

        # 添加消息到对话模板中，这里我们只添加图像部分而不包括问题
        conv.append_message(conv.roles[0], qs1)
        conv.append_message(conv.roles[1], None)

        # 获取最终的prompt1
        prompt1 = conv.get_prompt()

        # 使用tokenizer处理prompt1，并生成input_ids1
        input_ids1 = tokenizer_image_token(prompt1, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        # 获取两个张量的形状
        input_ids_shape = input_ids.shape[1]
        input_ids1_shape = input_ids1.shape[1]

        # 计算最长的序列长度
        max_seq_length = max(input_ids_shape, input_ids1_shape)

        # 获取填充标记的ID
        padding_value = tokenizer.pad_token_id

        # 对 input_ids 进行填充（如果需要）
        if input_ids_shape < max_seq_length:
            input_ids = torch.nn.functional.pad(input_ids, (max_seq_length - input_ids_shape, 0), value=padding_value)

        # 对 input_ids1 进行填充（如果需要）
        if input_ids1_shape < max_seq_length:
            input_ids1 = torch.nn.functional.pad(input_ids1, (max_seq_length - input_ids1_shape, 0), value=padding_value)



        # 拼接两个张量
        input_ids_combined = torch.cat((input_ids, input_ids1), dim=0)

        # test_input_ids = input_ids_combined[:1]  # 第一个样本
        # test_input_ids1 = input_ids_combined[1:2]  # 第二个样本

        # print(test_input_ids)
        # print(test_input_ids1)

    

        image = Image.open(os.path.join(args.image_folder, image_file))
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        
        if args.use_cd:
            image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
        else:
            image_tensor_cd = None      

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)


        with torch.inference_mode():
            output_ids = model.generate(
                input_ids_combined,
                image_file=image_file,
                images=image_tensor.unsqueeze(0).half().cuda(),
                images_cd=(image_tensor_cd.unsqueeze(0).half().cuda() if image_tensor_cd is not None else None),
                cd_alpha = args.cd_alpha,
                cd_beta = args.cd_beta,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_token,
                use_cache=True,
                use_IAVA=args.use_IAVA,
                layer_gamma=args.layer_gamma,
                masking_scheme=args.masking_scheme,
                lamb=args.lamb,
                use_m3id=args.use_m3id,
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        
        
        logger.info(f"[{image_file}]")
        logger.info(f"prompt: {cur_prompt}") 
        logger.info(f"text: {outputs}")  
        
        ## one word processing 
        outputs = recorder(outputs)          

        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "model_id": model_name,
                                   "image": image_file,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/sda/gaodh/projects/Prompt-Highlighter/checkpoints/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/sda/gaodh/projects/Prompt-Highlighter/base_models/LLaVA/playground/data/eval/MME/MME_Benchmark_release_version")
    parser.add_argument("--question-file", type=str, default="/sda/gaodh/projects/Prompt-Highlighter/base_models/LLaVA/playground/data/eval/MME/llava_mme.jsonl")
    parser.add_argument("--answers-file", type=str, default="/sda/gaodh/projects/IAVA/test/test1/answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    
    parser.add_argument("--log_path", type=str, default="/sda/gaodh/projects/IAVA/test/test1")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_cd", type=str2bool, default=False)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_IAVA", type=str2bool, default=True)
    parser.add_argument("--layer_gamma", type=float, default=0.5)
    parser.add_argument("--masking_scheme", type=str, default="zeros")
    parser.add_argument("--lamb", type=float, default=0)
    parser.add_argument("--max_token", type=int, default=64)
    parser.add_argument("--use_m3id", type=str2bool, default=False)

    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
