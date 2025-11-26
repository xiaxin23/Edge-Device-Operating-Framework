import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from models.vit import vit_lora, param_name_to_module_id_vit, vit_with_classifiers
from data import prepare_dataset
from tqdm import tqdm
from functools import partial
from copy import deepcopy
import argparse
import random
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed
)
from data.math_dataloader import load_hf_tokenizer, smart_tokenizer_and_embedding_resize,make_supervised_data_module
from loss_utils import ForCausalLMLoss, fixed_cross_entropy, ForCausalLMLoss_chunked
from models.pipeline_modeling_llama import llama_stage
from llm_profile import stage_partition
import time
from lora import LoRA
from  torch.distributed import rpc
import torch.multiprocessing as mp
import os
from parse_args import parse_args
from distributed_model import DistributedSubModel, register_rref, global_shared_rrefs
import torch.distributed as dist
from transformers import Trainer
import threading

# torch.cuda.set_device(9)

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

global vocab_size
global device
global remote_networks
global base_forward_futs

def watch_video_memory(device):
    allocated_memory = torch.cuda.memory_allocated(device)  # 已分配的显存
    reserved_memory = torch.cuda.memory_reserved(device)    # 已保留的显存
    print("  - 已分配的显存: {:.2f} GB".format(allocated_memory / (1024 ** 3)))
    print("  - 已保留的显存: {:.2f} GB".format(reserved_memory / (1024 ** 3)))

def tensor_abs_stats(tensor):
    # 计算绝对值平均值
    avg_abs = torch.mean(torch.abs(tensor))
    
    # 处理数量级众数
    abs_tensor = torch.abs(tensor)
    non_zero_mask = abs_tensor != 0
    non_zero_abs = abs_tensor[non_zero_mask]
    
    if non_zero_abs.numel() == 0:
        return avg_abs, None  # 或抛出异常
    
    # 计算数量级
    magnitudes = torch.floor(torch.log10(non_zero_abs)).long()
    values, counts = torch.unique(magnitudes, return_counts=True)
    mode_magnitude = values[torch.argmax(counts)]
    
    return avg_abs, mode_magnitude

def to_device(inputs, device):
    if isinstance(inputs, dict):
        output = {}
        for k, v in inputs.items():
            try:
                output[k] = v.to(device)
            except:
                output[k] = v
    else:
        output = []
        output = inputs.to(device)
    return output

def ours_set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def make_forward_callback(epoch_idx, iter_idx, micro_iter_idx, device):
    def _cb(prev_fut: torch.futures.Future, epoch_idx, iter_idx, micro_iter_idx, device):
        ctx = prev_fut.wait()
        return remote_networks[f"worker-{args.gpus[device]}"]\
            .rpc_async()\
            .base_forward_only(
                epoch_idx,
                iter_idx,
                micro_iter_idx,
            )
    return _cb
    
def main(args):
    #load tokenizer     
    # tokenizer = load_hf_tokenizer(args.model_name_or_path,
    #                             fast_tokenizer=True,
    #                             model_max_length=args.max_seq_len,
    #                             padding_side="right")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_seq_len,
        padding_side="right",
        use_fast=False,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    
    #load model    
    model_config = AutoConfig.from_pretrained(args.model_name_or_path)
    # with torch.device("meta"):
    network = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        # from_tf=bool(".ckpt" in args.model_name_or_path),
        config=model_config,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        # attn_implementation="eager"
    )
    #  resize
    print("*"*50)
    print("Before adding, tokenizer length: ",len(tokenizer))
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    print("special_tokens_dict: ",special_tokens_dict)
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=network,
    )
    print("*"*50)
    print("After adding, tokenizer length: ",len(tokenizer))
    args.vocab_size = len(tokenizer)
    
    args.micro_batch_num = int(args.batch_size//args.micro_batch_size)
    train_dataset, data_collator, train_dataloader = make_supervised_data_module(tokenizer, args)
    args.total_iterations = len(train_dataloader)
    
    model_split_configs, iszeroth_ls, device_to_stage_ls, stage_to_device_ls = stage_partition(args.training_methods, args.model_name_or_path)
    remote_networks = {}
    for device_idx, (model_split_config, iszeroth, device_to_stage) in enumerate(zip(model_split_configs, iszeroth_ls, device_to_stage_ls)):
        remote_networks[f"worker-{args.gpus[device_idx]}"] = rpc.remote(f"worker-{args.gpus[device_idx]}", DistributedSubModel, 
                                                             args=(args, network, device_idx, model_split_config, iszeroth, device_to_stage, device_to_stage_ls, iszeroth_ls))
    
    send_rref_signals = []
    for remote_name in list(remote_networks.keys()):
        for name, rref in remote_networks.items():
            if remote_name != name:
                # print("name: {}, rref: {}".format(name, rref))
                send_rref_signals.append(rpc.rpc_async(remote_name, register_rref, args=(name, rref)))
    for send_rref_signal in send_rref_signals:
        send_rref_signal.wait()

    # data_signals = []
    # for device_idx, model_split_config, iszeroth, device_to_stage in enumerate(zip(model_split_configs, iszeroth_ls, device_to_stage_ls)):
    #     if device_to_stage == 0 or device_to_stage==len(device_to_stage_ls)-1:
    #         data_signals.append(remote_networks[f"worker-{device_idx}"].rpc_async().set_supervised_data_module(tokenizer=tokenizer))
    # for data_signal in data_signals:
    #     data_signal.wait()
    
    #micro_batch trianing
    for epoch_idx in range(args.epochs):
        print('Epoch: {}, len(train_loaders): {}'.format(epoch_idx, len(train_dataloader)))
        for iter_idx, (inputs, labels) in enumerate(train_dataloader):
            micro_base_forward_signals = []
            micro_base_backward_signals = []
            start = time.time()
            
            micro_input_ids_ls = torch.split(inputs["input_ids"], args.micro_batch_size, dim=0)
            micro_attention_mask_ls = torch.split(inputs["attention_mask"], args.micro_batch_size, dim=0)
            micro_batch_datas_inputs = [
                dict(
                    input_ids=micro_input_ids,
                    attention_mask=micro_attention_mask,
                )
                for micro_input_ids, micro_attention_mask in zip(micro_input_ids_ls, micro_attention_mask_ls)
            ]
            micro_batch_datas_labels = torch.split(labels, args.micro_batch_size, dim=0)
            
            # set train configuration
            set_configuration_signals = []
            for stage_idx, device_idx in enumerate(stage_to_device_ls):
                set_configuration_signals.append(
                    remote_networks[f"worker-{args.gpus[device_idx]}"].rpc_async().set_train_configuration()  #DistributedSubModel.set_train_configuration
                )
            for set_configuration_signal in set_configuration_signals:
                set_configuration_signal.wait()
            for micro_iter_idx, labels in enumerate(micro_batch_datas_labels):
                # send labels to stage[-1] and last_zerothmodel
                receive_labels_signals = []
                receive_labels_signals.append(
                    remote_networks[f"worker-{args.gpus[stage_to_device_ls[-1]]}"].rpc_async().receive_labels(labels, micro_iter_idx)
                    )
                receive_labels_signals.append(
                    remote_networks[f"worker-{args.gpus[iszeroth_ls.count(True)-1]}"].rpc_async().receive_labels(labels, micro_iter_idx)
                    )
            for receive_labels_signal in receive_labels_signals:
                receive_labels_signal.wait()
            
            if args.training_methods == "only_zeroth":
                base_forward_total_steps = len(stage_to_device_ls) + args.micro_batch_num - 1
                base_forward_futs = [[None]*len(stage_to_device_ls) for _ in range(args.micro_batch_num)]
                for step_idx in range(base_forward_total_steps):
                    for stage_idx, device_idx in enumerate(stage_to_device_ls):
                        forward_mb_num = step_idx - stage_idx
                        if not (0 <= mb_num < args.micro_batch_num):
                            continue
                        if stage_idx == 0:
                            base_forward_futs[forward_mb_num][stage_idx] = \
                                remote_networks[f"worker-{args.gpus[stage_to_device_ls[0]]}"].rpc_async().base_forward_only(epoch_idx, 
                                                                                                                            iter_idx, micro_iter_idx, 
                                                                                                                            micro_batch_datas_inputs[forward_mb_num])
                        else:
                            # base_forward_futs[forward_mb_num][stage_idx] = base_forward_futs[forward_mb_num][stage_idx-1].then(
                            #         lambda epoch_idx=epoch_idx, iter_idx=iter_idx, stage_idx=stage_idx, forward_mb_num=forward_mb_num: 
                            #             remote_networks[f"worker-{args.gpus[stage_to_device_ls[stage_idx]]}"].rpc_async().base_forward_only(epoch_idx, iter_idx, forward_mb_num)
                            #     )
                            base_forward_futs[mb_num][stage_idx-1].wait()
                            base_forward_futs[mb_num][stage_idx] = \
                                remote_networks[f"worker-{args.gpus[stage_to_device_ls[stage_idx]]}"].rpc_async().base_forward_only(epoch_idx, 
                                                                                                                            iter_idx, micro_iter_idx)
                for mb_num in range(args.micro_batch_num):
                    base_forward_futs[mb_num][len(stage_to_device_ls)-1].wait()
            elif args.training_methods == "only_first":   #gpipe
                base_forward_total_steps = len(stage_to_device_ls) + args.micro_batch_num - 1
                base_backward_total_steps = len(stage_to_device_ls) + args.micro_batch_num - 1
                total_steps = 2 * (len(stage_to_device_ls) + args.micro_batch_num - 1)
                base_forward_futs = [[None]*len(stage_to_device_ls) for _ in range(args.micro_batch_num)]
                base_backward_futs = [[None]*len(stage_to_device_ls) for _ in range(args.micro_batch_num)]
                #forward
                for step_idx in range(total_steps):
                    if step_idx < base_forward_total_steps:
                        for stage_idx, device_idx in enumerate(stage_to_device_ls):
                            forward_mb_num = step_idx - stage_idx
                            if not (0 <= forward_mb_num < args.micro_batch_num):
                                continue
                            if stage_idx == 0:
                                base_forward_futs[forward_mb_num][0] = remote_networks[f"worker-{args.gpus[stage_to_device_ls[0]]}"].rpc_async().base_forward_only(epoch_idx, 
                                                                                                                                iter_idx, forward_mb_num, 
                                                                                                                                micro_batch_datas_inputs[forward_mb_num])
                            else:
                                # prev_fut = base_forward_futs[forward_mb_num][stage_idx-1]
                                # def _then_call(prev_res, *, epoch_idx=epoch_idx, iter_idx=iter_idx, forward_mb_num=forward_mb_num, stage_idx=stage_idx, device_idx=device_idx):
                                #     return remote_networks[f"worker-{args.gpus[stage_to_device_ls[stage_idx]]}"] \
                                #             .rpc_async().base_forward_only(epoch_idx, iter_idx, forward_mb_num)
                                # base_forward_futs[forward_mb_num][stage_idx] = prev_fut.then(_then_call)
                                base_forward_futs[forward_mb_num][stage_idx-1].wait()
                                base_forward_futs[forward_mb_num][stage_idx] = \
                                    remote_networks[f"worker-{args.gpus[stage_to_device_ls[stage_idx]]}"].rpc_async().base_forward_only(epoch_idx, iter_idx, forward_mb_num)
                    else:  #backward
                        for stage_idx, device_idx in enumerate(stage_to_device_ls):
                            inverse_stage = len(stage_to_device_ls) - 1 - stage_idx
                            backward_mb_num = step_idx - base_forward_total_steps - inverse_stage
                            if 0 <= backward_mb_num < args.micro_batch_num:
                                if inverse_stage == 0:
                                    base_forward_futs[backward_mb_num][len(stage_to_device_ls)-1].wait()
                                    base_backward_futs[backward_mb_num][inverse_stage] = \
                                        remote_networks[f"worker-{args.gpus[stage_to_device_ls[stage_idx]]}"].rpc_async().base_backward_only(backward_mb_num)
                                else:
                                    base_backward_futs[backward_mb_num][inverse_stage-1].wait()
                                    # print("backward_mb_num: ", backward_mb_num, "inverse_stage: ", inverse_stage, "waiting")
                                    base_backward_futs[backward_mb_num][inverse_stage] = \
                                        remote_networks[f"worker-{args.gpus[stage_to_device_ls[stage_idx]]}"].rpc_async().base_backward_only(backward_mb_num)
                        # print("-"*50)
                for backward_mb_num in range(args.micro_batch_num):
                    base_backward_futs[backward_mb_num][len(stage_to_device_ls)-1].wait()
            elif args.training_methods == "pipedream":
                base_forward_total_steps = len(stage_to_device_ls) + args.micro_batch_num - 1
                base_backward_total_steps = len(stage_to_device_ls) + args.micro_batch_num - 1
                total_steps = 2 * (len(stage_to_device_ls) + args.micro_batch_num - 1)
                base_forward_futs = [[None]*len(stage_to_device_ls) for _ in range(args.micro_batch_num)]
                base_backward_futs = [[None]*len(stage_to_device_ls) for _ in range(args.micro_batch_num)]
                for step_idx in range(total_steps):
                    if step_idx < len(stage_to_device_ls):
                        for stage_idx, device_idx in enumerate(stage_to_device_ls):
                            forward_mb_num = step_idx - stage_idx
                            if 0 <= forward_mb_num < args.micro_batch_num:
                                if stage_idx == 0:
                                    base_forward_futs[forward_mb_num][stage_idx] = \
                                        remote_networks[f"worker-{args.gpus[stage_to_device_ls[0]]}"].rpc_async().base_forward_only(epoch_idx,
                                                                                                                                    iter_idx, forward_mb_num,
                                                                                                                                    micro_batch_datas_inputs[forward_mb_num])
                                else:
                                    base_forward_futs[forward_mb_num][stage_idx-1].wait()
                                    base_forward_futs[forward_mb_num][stage_idx] = \
                                        remote_networks[f"worker-{args.gpus[stage_to_device_ls[stage_idx]]}"].rpc_async().base_forward_only(epoch_idx, iter_idx, forward_mb_num)
                    else:
                        for stage_idx, device_idx in enumerate(stage_to_device_ls):
                            forward_mb_num, backward_mb_num = None, None
                            inverse_stage = len(stage_to_device_ls) - 1 - stage_idx
                            if ((step_idx - stage_idx) % 2) == 0:
                                forward_mb_num = (step_idx - stage_idx) // 2
                            else:
                                backward_mb_num = ((step_idx - (len(stage_to_device_ls)-1)) - inverse_stage) // 2
                            
                            if (forward_mb_num is not None) and (inverse_stage < forward_mb_num < args.micro_batch_num):
                                # print("fowarad satge:{}, micro-batch {}".format(stage_idx, forward_mb_num))
                                if stage_idx == 0:
                                    base_forward_futs[forward_mb_num][stage_idx] = \
                                        remote_networks[f"worker-{args.gpus[stage_to_device_ls[0]]}"].rpc_async().base_forward_only(epoch_idx,
                                                                                                                                    iter_idx, forward_mb_num,
                                                                                                                                    micro_batch_datas_inputs[forward_mb_num])
                                else:
                                    base_forward_futs[forward_mb_num][stage_idx-1].wait()
                                    base_forward_futs[forward_mb_num][stage_idx] = \
                                        remote_networks[f"worker-{args.gpus[stage_to_device_ls[stage_idx]]}"].rpc_async().base_forward_only(epoch_idx, iter_idx, forward_mb_num)
                            elif (backward_mb_num is not None) and (0 <= backward_mb_num < args.micro_batch_num):
                                # print("backward satge:{}, micro-batch {}".format(stage_idx, backward_mb_num))
                                if inverse_stage == 0:
                                    base_forward_futs[backward_mb_num][len(stage_to_device_ls)-1].wait()
                                    base_backward_futs[backward_mb_num][inverse_stage] = \
                                        remote_networks[f"worker-{args.gpus[stage_to_device_ls[stage_idx]]}"].rpc_async().base_backward_only(backward_mb_num)
                                else:
                                    base_backward_futs[backward_mb_num][inverse_stage-1].wait()
                                    base_backward_futs[backward_mb_num][inverse_stage] = \
                                        remote_networks[f"worker-{args.gpus[stage_to_device_ls[stage_idx]]}"].rpc_async().base_backward_only(backward_mb_num)
                for backward_mb_num in range(args.micro_batch_num):
                    base_backward_futs[backward_mb_num][len(stage_to_device_ls)-1].wait()
            elif args.training_methods == "hybrid": 
                estimate_total_steps = max(args.micro_batch_num + args.q_num * args.micro_batch_num + iszeroth_ls.count(True) - 1, 
                                           2*(len(stage_to_device_ls) + args.micro_batch_num - 1)
                                           )
                base_forward_total_steps = len(stage_to_device_ls) + args.micro_batch_num - 1
                base_backward_total_steps = len(stage_to_device_ls) + args.micro_batch_num - 1
                base_forward_futs = [[None]*len(stage_to_device_ls) for _ in range(args.micro_batch_num)]
                base_backward_futs = [[None]*iszeroth_ls.count(False) for _ in range(args.micro_batch_num)]
                estimate_forward_futs = [[[None]*iszeroth_ls.count(True) for _ in range(args.q_num)] for _ in range(args.micro_batch_num)]
                
                for step_idx in range(estimate_total_steps):
                    if step_idx < len(stage_to_device_ls):
                        for stage_idx, device_idx in enumerate(stage_to_device_ls):
                            forward_mb_num = step_idx - stage_idx
                            if 0 <= forward_mb_num < args.micro_batch_num:
                                print("fowarad satge:{}, micro-batch {}".format(stage_idx, forward_mb_num))
                                if stage_idx == 0:
                                    base_forward_futs[forward_mb_num][stage_idx] = \
                                        remote_networks[f"worker-{args.gpus[stage_to_device_ls[0]]}"].rpc_async().base_forward_only(epoch_idx,
                                                                                                                                    iter_idx, forward_mb_num,
                                                                                                                                    micro_batch_datas_inputs[forward_mb_num])
                                else:
                                    base_forward_futs[forward_mb_num][stage_idx-1].wait()
                                    base_forward_futs[forward_mb_num][stage_idx] = \
                                        remote_networks[f"worker-{args.gpus[stage_to_device_ls[stage_idx]]}"].rpc_async().base_forward_only(epoch_idx, iter_idx, forward_mb_num)
                    else:
                        if step_idx <= max(args.micro_batch_num, len(stage_to_device_ls)):
                            for stage_idx, iszeroth in enumerate(iszeroth_ls):
                                if not iszeroth:
                                    break
                                forward_mb_num = step_idx - stage_idx
                                if 0 <= forward_mb_num < args.micro_batch_num and iszeroth_ls[stage_idx]:
                                    print("fowarad satge:{}, micro-batch {}".format(stage_idx, forward_mb_num))
                                    if stage_idx == 0:
                                        base_forward_futs[forward_mb_num][stage_idx] = \
                                            remote_networks[f"worker-{args.gpus[stage_to_device_ls[0]]}"].rpc_async().base_forward_only(epoch_idx,
                                                                                                                                        iter_idx, forward_mb_num,
                                                                                                                                        micro_batch_datas_inputs[forward_mb_num])
                                    else:
                                        base_forward_futs[forward_mb_num][stage_idx-1].wait()
                                        base_forward_futs[forward_mb_num][stage_idx] = \
                                            remote_networks[f"worker-{args.gpus[stage_to_device_ls[stage_idx]]}"].rpc_async().base_forward_only(epoch_idx, iter_idx, forward_mb_num)
                        
                        for stage_idx, device_idx in enumerate(stage_to_device_ls):
                            if not iszeroth_ls[stage_idx]:
                                forward_mb_num, backward_mb_num = None, None
                                inverse_stage = len(stage_to_device_ls) - 1 - stage_idx
                                if ((step_idx - stage_idx) % 2) == 0:
                                    forward_mb_num = (step_idx - stage_idx) // 2
                                else:
                                    backward_mb_num = ((step_idx - (len(stage_to_device_ls)-1)) - inverse_stage) // 2
                                
                                if (forward_mb_num is not None) and (inverse_stage < forward_mb_num < args.micro_batch_num):
                                    print("fowarad satge:{}, micro-batch {}".format(stage_idx, forward_mb_num))
                                    if stage_idx == 0:
                                        base_forward_futs[forward_mb_num][stage_idx] = \
                                            remote_networks[f"worker-{args.gpus[stage_to_device_ls[0]]}"].rpc_async().base_forward_only(epoch_idx,
                                                                                                                                        iter_idx, forward_mb_num,
                                                                                                                                        micro_batch_datas_inputs[forward_mb_num])
                                    else:
                                        base_forward_futs[forward_mb_num][stage_idx-1].wait()
                                        base_forward_futs[forward_mb_num][stage_idx] = \
                                            remote_networks[f"worker-{args.gpus[stage_to_device_ls[stage_idx]]}"].rpc_async().base_forward_only(epoch_idx, iter_idx, forward_mb_num)
                                elif (backward_mb_num is not None) and (0 <= backward_mb_num < args.micro_batch_num):
                                    print("backward satge:{}, micro-batch {}".format(stage_idx, backward_mb_num))
                                    if inverse_stage == 0:
                                        base_forward_futs[backward_mb_num][len(stage_to_device_ls)-1].wait()
                                        base_backward_futs[backward_mb_num][inverse_stage] = \
                                            remote_networks[f"worker-{args.gpus[stage_to_device_ls[stage_idx]]}"].rpc_async().base_backward_only(backward_mb_num)
                                    else:
                                        base_backward_futs[backward_mb_num][inverse_stage-1].wait()
                                        base_backward_futs[backward_mb_num][inverse_stage] = \
                                            remote_networks[f"worker-{args.gpus[stage_to_device_ls[stage_idx]]}"].rpc_async().base_backward_only(backward_mb_num)
                    #estimate
                    for stage_idx, iszeroth in enumerate(iszeroth_ls):
                        if not iszeroth:
                            break
                        estimate_mb_num = (step_idx - stage_idx - args.micro_batch_num) // args.q_num
                        estimate_q_cnt = (step_idx - stage_idx - args.micro_batch_num) % args.q_num
                        if (0 <= estimate_mb_num < args.micro_batch_num):
                            print("forward satge:{}, micro-batch {} estimate_q_cnt {}".format(stage_idx, estimate_mb_num, estimate_q_cnt))
                            if stage_idx == 0:
                                if estimate_q_cnt == 0:
                                    estimate_forward_futs[estimate_mb_num][estimate_q_cnt][stage_idx] = \
                                        remote_networks[f"worker-{args.gpus[stage_to_device_ls[stage_idx]]}"].rpc_async().estimate_grads(epoch_idx, 
                                                                                                                                         iter_idx, estimate_mb_num, estimate_q_cnt,
                                                                                                                                         micro_batch_datas_inputs[estimate_mb_num])
                                else:
                                    estimate_forward_futs[estimate_mb_num][estimate_q_cnt][stage_idx] = \
                                        remote_networks[f"worker-{args.gpus[stage_to_device_ls[stage_idx]]}"].rpc_async().estimate_grads(epoch_idx, 
                                                                                                                                iter_idx, estimate_mb_num, estimate_q_cnt)
                            else:
                                estimate_forward_futs[estimate_mb_num][estimate_q_cnt][stage_idx-1].wait()
                                estimate_forward_futs[estimate_mb_num][estimate_q_cnt][stage_idx] = \
                                    remote_networks[f"worker-{args.gpus[stage_to_device_ls[stage_idx]]}"].rpc_async().estimate_grads(epoch_idx, 
                                                                                                                             iter_idx, estimate_mb_num, estimate_q_cnt)
                    print("-"*50)
                for mb_num in range(args.micro_batch_num):
                    base_backward_futs[mb_num][iszeroth_ls.count(False)-1].wait()
                for mb_num in range(args.micro_batch_num):
                    for q_cnt in range(args.q_num):
                        estimate_forward_futs[mb_num][q_cnt][iszeroth_ls.count(True)-1].wait()

                remote_networks[f"worker-{args.gpus[stage_to_device_ls[-1]]}"].rpc_sync().send_base_loss()
            
            #optimizer
            optimizer_signals = []
            for stage_idx, device_idx in enumerate(stage_to_device_ls):
                optimizer_signals.append(
                    remote_networks[f"worker-{args.gpus[device_idx]}"].rpc_async().network_optimzier()
                )
            for optimizer_signal in optimizer_signals:
                optimizer_signal.wait()

            end = time.time()
            print("Iter: {}, time: {}".format(iter_idx, end-start))

def init_process(rank, world_size, args):
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    print("rank: ",rank)
    
    if rank == 0:
        rpc.init_rpc(f"master", rank=rank, world_size=world_size)
        main(args)
    else:
        gpu = args.gpus[(rank-1)]
        rpc.init_rpc(f"worker-{gpu}", rank=rank, world_size=world_size)
    rpc.shutdown()
    

if __name__ == "__main__":
    args = parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    ours_set_seed(args.seed)
    set_seed(args.seed)
    # device = f"cuda:{args.device}"
    
    args.gpus = args.gpus.split(',')
    world_size = 1 + len(args.gpus)
    # dist.init_process_group(backend="nccl")
    mp.spawn(init_process, args=(world_size, args), nprocs=world_size, join=True)
