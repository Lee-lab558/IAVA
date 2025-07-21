import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
import random
import numpy as np
import torch.backends.cudnn as cudnn
import torch.distributed as dist

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/experiments')
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ['CUDA_VISIBLE_DEVICES'] = '8'

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from utils import dist_util
from utils.logger import create_logger
from glob import glob

from PIL import Image
import math

from pope_loader import POPEDataSet

# import kornia
from transformers import set_seed
from IAVA_utils.vcd_add_noise import add_diffusion_noise
from IAVA_utils.IAVA_sample import evolve_IAVA_sampling
evolve_IAVA_sampling()

torch.multiprocessing.set_sharing_strategy('file_system')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description="POPE-Adv evaluation on LVLMs.")
    parser.add_argument("--model-path", type=str, default="/sda/gaodh/projects/Prompt-Highlighter/checkpoints/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=None)
    
    parser.add_argument("--pope_path", type=str, default="/sda/gaodh/projects/Prompt-Highlighter/base_models/LLaVA/playground/data/eval/pope/coco/coco_pope_random.json")
    parser.add_argument("--data_path", type=str, default="/sda/gaodh/projects/Prompt-Highlighter/base_models/LLaVA/playground/data/eval/pope/val2014")
    parser.add_argument("--log_path", type=str, default="/sda/gaodh/projects/IAVA/pope/test8/random")

    parser.add_argument("--noise_step", type=int, default=500)
    parser.add_argument("--use_cd", type=str2bool, default=False)
    parser.add_argument("--cd_alpha", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=int, default=7, help="specify the gpu to load the model.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--use_IAVA", type=str2bool, default=True)
    parser.add_argument("--layer_gamma", type=float, default=0.5)
    parser.add_argument("--masking_scheme", type=str, default="zeros")
    parser.add_argument("--lamb", type=float, default=0.99)
    parser.add_argument("--exp_description", type=str, default="..")
    parser.add_argument("--max_token", type=int, default=64)

    parser.add_argument("--cd_beta", type=float, default=0.1)

    args = parser.parse_args()
    return args


def setup_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def print_acc(pred_list, label_list):
    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)
    # unknown_ratio = pred_list.count(2) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))
    return acc, precision, recall, f1, yes_ratio

def recorder(out, pred_list):
    NEG_WORDS = ["No", "not", "no", "NO"]
    for line in out:
        line = line.replace('.', '')
        line = line.replace(',', '')
        words = line.split(' ')
        if any(word in NEG_WORDS for word in words) or any(word.endswith("n't") for word in words):
            pred_list.append(0)
        else:
            pred_list.append(1)
    
    return pred_list

def main():
    args = parse_args()


    # Setup DDP:
    dist_util.setup_dist(args)
    device = dist_util.device()

    # Setup an experiment folder:
    if dist.get_rank() == 0:
        os.makedirs(
            args.log_path, exist_ok=True
        )  # Make results folder (holds all experiment subfolders)
        model_string_name = args.model_path.split("/")[-1]
        experiment_dir = f"{args.log_path}/{model_string_name}/IAVA_{args.use_IAVA}_IAVA_{args.use_cd}_cd_alpha_${args.cd_alpha}"  # Create an experiment folder
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info(f"exp_description: {args.exp_description}")
    else:
        logger = create_logger(None)

    # ========================================
    #             Model Initialization
    # ========================================
    print('Initializing Model')
    logger.info(f"use_cd: {args.use_cd}, method: {args.use_IAVA}")
    logger.info(f"pope_path: {args.pope_path}")
    logger.info(f"alpha = {args.cd_alpha}, beta = {args.cd_beta}, max_token = {args.max_token}")

    
    #### for IAVA
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    
    tokenizer.padding_side = "left" 
    # load pope data
    # pope_path : json path, e.g. data/POPE/coco/coco_pope_random.json
    # data_path : image folder path e.g. data/coco/images/val2024
    
    pope_dataset = POPEDataSet(
        pope_path=args.pope_path, 
        data_path=args.data_path,
        trans=image_processor,
        model='llava'
    )
    pope_loader = torch.utils.data.DataLoader(
        pope_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        drop_last=False
    )

    print ("load data finished")

    print("Start eval...")
    pred_list, label_list = [], []
    for batch_id, data in tqdm(enumerate(pope_loader), total=len(pope_loader)):

        image = data["image"]
        qs = data["query"]
        qs1 = ["Please describe the content of the picture."]
        label = data["label"]
        image_path = data["image_path"]
        label_list = label_list + list(label)

        # ==============================================
        #             Text prompt setting
        # ==============================================
        
        if model.config.mm_use_im_start_end:
            qu = [DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + _ for _ in qs]
        else:
            qu = [DEFAULT_IMAGE_TOKEN + '\n' + _ for _ in qs]

        if model.config.mm_use_im_start_end:
            qu1 = [DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + _ for _ in qs1]
        else:
            qu1 = [DEFAULT_IMAGE_TOKEN + '\n' + _ for _ in qs1]
        
        input_ids = []
        input_ids1 = []

        for i in range(args.batch_size):
            conv = conv_templates[args.conv_mode].copy() 
            conv.append_message(conv.roles[0], qu[i] )
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            conv = conv_templates[args.conv_mode].copy() 
            conv.append_message(conv.roles[0], qu1[i] )
            conv.append_message(conv.roles[1], None)
            prompt1 = conv.get_prompt()
            
        # ==============================================
        #             Image tensor setting
        # ==============================================
            
            input_id = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            input_id1 = tokenizer_image_token(prompt1, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            input_ids.append(input_id)
            input_ids1.append(input_id1)

            
        def make_batch(input_ids):
            input_ids = [_.squeeze(0) for _ in input_ids]
            max_len = max([_.shape[0] for _ in input_ids])
            input_ids = [torch.cat([torch.zeros(max_len - _.shape[0], dtype=torch.long).cuda(), _], dim=0) for _ in input_ids]
            return torch.stack(input_ids, dim=0)
        
        input_ids = make_batch(input_ids)
        input_ids1 = make_batch(input_ids1)

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
        
        image_tensor = image

        # ==============================================
        #             IAVA method setting
        # ==============================================
        if args.use_cd:
            image_tensor_cd = add_diffusion_noise(image_tensor, noise_step=500)
        else:
            image_tensor_cd = None    
        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
        with torch.inference_mode():
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids_combined,
                    images=image_tensor.half().cuda(),
                    images_cd=(image_tensor_cd.half().cuda() if image_tensor_cd is not None else None),
                    cd_alpha=args.cd_alpha,
                    cd_beta=args.cd_beta,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_new_tokens=args.max_token,
                    use_cache=True,
                    use_IAVA=args.use_IAVA,
                    layer_gamma=args.layer_gamma,
                    masking_scheme=args.masking_scheme,
                    lamb=args.lamb,
                    temp=args.temperature,
                )
                
                input_token_len = input_ids.shape[1]
                n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
                outputs = [_.strip() for _ in outputs]
                outputs = [_[:-len(stop_str)] if _.endswith(stop_str) else _ for _ in outputs]
                pred_list = recorder(outputs, pred_list)

                for ip, q, a, gt in zip(image_path, qs, outputs, label):
                    logger.info(f"[{ip}]")
                    logger.info(f"Q: {q}")
                    logger.info(f"A: {a}")
                    if gt == 1: logger.info(f"GT: Yes")
                    elif gt == 0: logger.info(f"GT: No")


    print("[{}]===============================================".format(args.seed))
    if len(pred_list) != 0:
        acc, precision, recall, f1, yes_ratio = print_acc(pred_list, label_list)
        logger.info(
            f"acc: {acc}, precision: {precision}, recall: {recall}, f1: {f1}, yes_ratio: {yes_ratio}"
        )
        

if __name__ == "__main__":
    main()
