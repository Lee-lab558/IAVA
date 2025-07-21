import copy
import argparse
import inspect
import warnings
import cv2
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import os
import torch
import torch.distributed as dist
from torch import nn
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties

from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import transformers
from transformers.generation.utils import SampleOutput
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM

from PIL import Image
import numpy as np
import sys

from llava.mm_utils import process_images


import numpy as np
import os
from PIL import Image, ImageFilter


def sample(
    self,
    input_ids_combined: torch.LongTensor,
    # images,
    # input_ids: torch.LongTensor,  #  暂定
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    use_IAVA : Optional[bool] = True,
    use_m3id : Optional[bool] = False,
    use_cd : Optional[bool] = False,
    **model_kwargs,
) -> Union[SampleOutput, torch.LongTensor]:
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id


    # 获取 input_ids 和 input_ids1
    input_ids = input_ids_combined[:1]  # 第一个样本
    input_ids1 = input_ids_combined[1:2]  # 第二个样本


    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )

    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    this_peer_finished = False  # used by synced_gpus only

    # auto-regressive generation
    model_kwargs_method = copy.deepcopy(model_kwargs)
    model_kwargs_cd = copy.deepcopy(model_kwargs)
    model_kwargs_m3id = copy.deepcopy(model_kwargs)
    t = 1
    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tesnsor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # 准备输入llava
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # 准备输入instructblip
        # model_inputs, _ = self.prepare_inputs_for_generation(input_ids, input_ids1, **model_kwargs)


        output_attentions = use_IAVA and not (model_kwargs.get("use_cache") and model_kwargs.get("past_key_values") is not None)

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states, 
        )

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]
        
        ## For contrastive decoding initial
        use_cd = model_kwargs.get("images_cd") != None
        output_attentions_wo_img = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states_wo_img = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )

        
        if use_IAVA:
            ## analyzing attetion logit
            layer_gamma = model_kwargs.get("layer_gamma")
            masking_scheme = model_kwargs.get("masking_scheme")
            model_name = model_kwargs.get("model_name") if model_kwargs.get("model_name") is not None else "llava"
    
            def count_top_p(img_att_logit, top_p=0.8):
                    """
                    img_att_logit: torch.Tensor, shape (1, N)
                    """
                    norm_img_att_logit = img_att_logit / img_att_logit.sum()
                    sorted_img_att_logit = torch.sort(norm_img_att_logit, descending=True)[0]
                    
                    return (torch.cumsum(sorted_img_att_logit, dim=1) < top_p).sum() + 1
            
            
    
            # 准备输入 llava
            # 准备仅包含图像输入的模型输入
            model_inputs_method = self.prepare_inputs_for_generation_method(input_ids, **model_kwargs_method)
            model_inputs_image_only = self.prepare_inputs_for_generation_method(input_ids1, **model_kwargs_method)

            # 准备输入instructblip
            # model_inputs_method, model_inputs_image_only = self.prepare_inputs_for_generation_method(input_ids, input_ids1, **model_kwargs_method)


            mask_idx = None
            if model_inputs_method.get("past_key_values") is None:
                attention = outputs.attentions             
                
                if model_name.lower() == "llava":
                    img_idx = list(range(35, 35+576))

                elif model_name.lower() == "blip":
                    img_idx = list(range(32))
                    
                layer_img_att_portion = []
                for logit in outputs.attentions:
                    img_logit = logit.mean(dim=1)[:,-1, img_idx]
                    layer_img_att_portion.append(img_logit.sum())
                    
                layer_img_att_portion = torch.stack(layer_img_att_portion, dim=0)
                total_img_att_portion = layer_img_att_portion.sum()
                layer_img_att_portion = layer_img_att_portion / total_img_att_portion
                k = count_top_p(layer_img_att_portion.unsqueeze(0), top_p=float(layer_gamma))
                
                _, top_k_lay_idx = torch.topk(layer_img_att_portion.float(), k, dim=0)

    
                # 第一次注意力计算：仅输入图像
                # 执行注意力计算，并按照选定层处理注意力
                outputs_image_only = self(**model_inputs_image_only, return_dict=True, output_attentions=True)
                att_logits_without_text = torch.stack(
                    [outputs_image_only.attentions[i].mean(dim=1)[:, -1, img_idx] for i in top_k_lay_idx],
                    dim=1  # 将选定层的注意力按照维度1堆叠
                    )

                

                # 第二次注意力计算：输入图像和问题文本
                outputs_with_text = self(**model_inputs_method, return_dict=True, output_attentions=True)
                att_logits_with_text = torch.stack(
                    [outputs_with_text.attentions[i].mean(dim=1)[:, -1, img_idx] for i in top_k_lay_idx],
                    dim=1  # 同样按照维度1堆叠
                    )

                # 计算两次注意力的变化
                attention_changes = att_logits_with_text - att_logits_without_text

                # 计算输入问题文本后注意力平均值
                avg_attention_with_text = att_logits_with_text.mean(dim=1)

                # 平均变化
                avg_changes = attention_changes.mean(dim=1)

                # 将带有问题文本的注意力权重进行排序
                sorted_attention_with_text,_ = torch.sort(avg_attention_with_text.view(-1))
                # 将排序后的张量转换为NumPy数组
                sorted_attention_arry = sorted_attention_with_text.cpu().numpy()
                # threshold_attention = sorted_attention_arry[10]

                # 将 avg_changes 转换为一维，并按从小到大排序
                sorted_changes, _ = torch.sort(avg_changes.view(-1))
                # 将排序后的张量转换为 NumPy 数组
                sorted_array = sorted_changes.cpu().numpy()

                # avg_changes_abs = torch.abs(avg_changes)
                # 确定盲标记
                # 不相关标记，意味着输入查询之后，注意力得分会下降，因此阈值应设置的较小，保证不相关标记的数量较少，因为这些标记占据了
                # 大部分注意力得分，会影响回答，所以应当屏蔽他们
                threshold_change = sorted_array[16] # 根据模型的不同，需要调整阈值
                blind_mask = torch.where((avg_changes < 0)&(avg_changes < threshold_change)&(avg_attention_with_text.float() > sorted_attention_arry.mean()-0*sorted_attention_arry.std()), 0, 1)
                # blind_mask = torch.where(avg_changes < threshold_change, 0, 1)

                # 应用盲标记（0）处理,把非盲标记屏蔽，对应变化大的，对应blind_mask值为1的
                mask_idx = torch.where(blind_mask > 0)[1].unsqueeze(0)

                # 可视化展示    
                # apply_mask_and_save_single(image_file=model_kwargs['image_file'], mask_idx=mask_idx, img_idx=img_idx, image_processor=self.image_processor,model_config=self.config)
    
            # 更新模型输入
            output_attentions_method = False
            model_inputs_method.update(
                { 
                    "mask_idx": mask_idx,
                    "masking_scheme": masking_scheme
                }
            )
    
            # 生成输出
            outputs_method = self(
                **model_inputs_method,
                return_dict=True,
                output_attentions=output_attentions_method,
                output_hidden_states=output_hidden_states,
            )
    
            # 处理下一步的预测
            next_token_logits_method = outputs_method.logits[:, -1, :]
    
            if torch.isnan(next_token_logits_method).any():
                next_token_logits_method = next_token_logits
    
            IAVA_alpha = model_kwargs.get("cd_alpha") if model_kwargs.get("cd_alpha") is not None else 1.0
            IAVA_beta = model_kwargs.get("cd_beta") if model_kwargs.get("cd_beta") is not None else 0.1
    
            cutoff = torch.log(torch.tensor(IAVA_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
            diffs = (1+IAVA_alpha)*next_token_logits - IAVA_alpha*next_token_logits_method
            IAVA_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))
            IAVA_logits = logits_processor(input_ids, IAVA_logits)
            IAVA_logits = logits_warper(input_ids, IAVA_logits)
    
            next_token_scores = IAVA_logits
            IAVA_probs = nn.functional.softmax(IAVA_logits, dim=-1)
            next_tokens = torch.multinomial(IAVA_probs, num_samples=1).squeeze(1)       
        
        elif use_cd:
            ## cd_comments: forward pass of the model with distorted image input
            model_inputs_cd = self.prepare_inputs_for_generation_cd(input_ids, **model_kwargs_cd)
            outputs_cd = self(
                **model_inputs_cd,
                return_dict=True,
                output_attentions=output_attentions_wo_img,
                output_hidden_states=output_hidden_states_wo_img,
            )
            next_token_logits_cd = outputs_cd.logits[:, -1, :]
            
            ## cd_comments: pre-process logits from contrastive inputs
            cd_alpha = model_kwargs.get("cd_alpha") if model_kwargs.get("cd_alpha") is not None else 0.5
            cd_alpha = 1.0
            cd_beta = model_kwargs.get("cd_beta") if model_kwargs.get("cd_beta") is not None else 0.1

            # version 2 set cutoff for Adaptive Plausibility Constraints
            cutoff = torch.log(torch.tensor(cd_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
            
            diffs = (1+cd_alpha)*next_token_logits - cd_alpha*next_token_logits_cd
            cd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))

            ## cd_comments: apply temperature warping and top-k filtering in contrastive decoding
            cd_logits = logits_processor(input_ids, cd_logits)
            cd_logits = logits_warper(input_ids, cd_logits)

            next_token_scores = cd_logits
            cd_probs = nn.functional.softmax(cd_logits, dim=-1)
            next_tokens = torch.multinomial(cd_probs, num_samples=1).squeeze(1)

        elif use_m3id:
            import math
            lamda = 0.02
            gamma_t = math.exp(-lamda * t)
            t += 1
            
            model_inputs_m3id = self.prepare_inputs_for_generation_m3id(input_ids, **model_kwargs_m3id)
            outputs_m3id = self(
                **model_inputs_m3id,
                return_dict=True,
                output_attentions=output_attentions_wo_img,
                output_hidden_states=output_hidden_states_wo_img,
            )
            next_token_logits_m3id = outputs_m3id.logits[:, -1, :]
            
            cd_beta = model_kwargs.get("cd_beta") if model_kwargs.get("cd_beta") is not None else 0.1

            # version 2 set cutoff for Adaptive Plausibility Constraints
            cutoff = torch.log(torch.tensor(cd_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
            
            lc = torch.log_softmax(next_token_logits, dim=-1)
            lu = torch.log_softmax(next_token_logits_m3id, dim=-1)
            m3id_logit = lc + ((1-gamma_t)/gamma_t)*(lc - lu)
            m3id_logit = m3id_logit.masked_fill(next_token_logits < cutoff, -float("inf"))
            
            
            m3id_logit = logits_processor(input_ids, m3id_logit)
            m3id_logit = logits_warper(input_ids, m3id_logit)
            
            next_token_scores = m3id_logit
            m3id_probs = nn.functional.softmax(m3id_logit, dim=-1)
            next_tokens = torch.multinomial(m3id_probs, num_samples=1).squeeze(1)
        else:
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)
            next_token_scores = next_token_scores
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            
        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )


        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        if use_IAVA:
            model_kwargs_method = self._update_model_kwargs_for_generation(
                outputs_method, model_kwargs_method, is_encoder_decoder=self.config.is_encoder_decoder
            )
        ## cd_comments: update model_kwargs_cd for contrastive decoding
        if use_cd:
            model_kwargs_cd = self._update_model_kwargs_for_generation(
                outputs_cd, model_kwargs_cd, is_encoder_decoder=self.config.is_encoder_decoder
            )
        if use_m3id:
            model_kwargs_cd = self._update_model_kwargs_for_generation(
                outputs_m3id, model_kwargs_m3id, is_encoder_decoder=self.config.is_encoder_decoder
            )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return SampleEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return SampleDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return input_ids

def evolve_IAVA_sampling():
    transformers.generation.utils.GenerationMixin.sample = sample

