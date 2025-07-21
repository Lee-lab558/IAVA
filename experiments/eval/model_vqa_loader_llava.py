import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '8'



sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria
from torch.utils.data import Dataset, DataLoader

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

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["image"]
        qs = line["text"]
        qs1 = "Please describe the content of the picture."

        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        if self.model_config.mm_use_im_start_end:
            qs1 = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs1
        else:
            qs1 = DEFAULT_IMAGE_TOKEN + '\n' + qs1

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs1)
        conv.append_message(conv.roles[1], None)
        prompt1 = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids1 = tokenizer_image_token(prompt1, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        if args.use_cd:
            image_tensor_cd = add_diffusion_noise(image_tensor, args.noise_step)
        else:
            image_tensor_cd = None      
        

        return input_ids, input_ids1, image_tensor,image_tensor_cd

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder, tokenizer, image_processor, model.config)

    for (input_ids, input_ids1, image_tensor, image_tensor_cd), line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        cur_prompt = line["text"]

        input_ids = input_ids.to(device='cuda', non_blocking=True)
        input_ids1 = input_ids1.to(device='cuda', non_blocking=True)

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

        # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        # keywords = [stop_str]
        # stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids_combined,
                images=image_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True),
                images_cd=(image_tensor_cd.unsqueeze(0).half().cuda() if image_tensor_cd is not None else None),
                # image_tensor_cd=image_tensor_cd.to(dtype=torch.float16, device='cuda', non_blocking=True),
                cd_alpha = args.cd_alpha,
                cd_beta = args.cd_beta,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                use_IAVA=args.use_IAVA,
                layer_gamma=args.layer_gamma,
                masking_scheme=args.masking_scheme,
                lamb=args.lamb,
                use_m3id=args.use_m3id,
                use_cd=args.use_cd,
                )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/sda/gaodh/projects/Prompt-Highlighter/checkpoints/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/sda/gaodh/projects/Prompt-Highlighter/base_models/LLaVA/playground/data/eval/textvqa/train_images")
    parser.add_argument("--question-file", type=str, default="/sda/gaodh/projects/Prompt-Highlighter/base_models/LLaVA/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl")
    parser.add_argument("--answers-file", type=str, default="/sda/gaodh/projects/IAVA/textvqa_answers/test4/test4.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_cd", type=str2bool, default=True)
    parser.add_argument("--cd_alpha", type=float, default=1)
    parser.add_argument("--cd_beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--use_IAVA", type=str2bool, default=False)
    parser.add_argument("--layer_gamma", type=float, default=0.5)
    parser.add_argument("--masking_scheme", type=str, default="zeros")
    parser.add_argument("--lamb", type=float, default=0)
    parser.add_argument("--use_m3id", type=str2bool, default=False)
    args = parser.parse_args()

    set_seed(args.seed)
    eval_model(args)
