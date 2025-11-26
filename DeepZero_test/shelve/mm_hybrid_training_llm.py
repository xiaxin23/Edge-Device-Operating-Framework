import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
# from models.vit import vit_lora, param_name_to_module_id_vit, vit_with_classifiers
# from data import prepare_dataset
from tqdm import tqdm
from functools import partial
from copy import deepcopy
import argparse
import random
# from peft import (
#     LoraConfig,
#     PeftModel,
#     get_peft_model,
#     get_peft_model_state_dict,
#     prepare_model_for_kbit_training,
#     set_peft_model_state_dict,
# )
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
import socket
import fcntl
import struct
from datetime import timedelta
from collections import defaultdict
import functools
from torch.futures import collect_all

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

def set_all_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def get_ip_address(ifname):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(
        fcntl.ioctl(
            s.fileno(),
            0x8915,  # SIOCGIFADDR
            struct.pack('256s', ifname[:15].encode('utf-8'))
        )[20:24]
    )

def load_train_dataloader(args,):
    #load tokenizer     
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/nvidia/Llama-3.2-3B_old/",
        model_max_length=args.max_seq_len,
        padding_side="right",
        use_fast=False,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    
    #load model    
    model_config = AutoConfig.from_pretrained("/home/nvidia/Llama-3.2-3B_old/")
    # with torch.device("meta"):
    network = AutoModelForCausalLM.from_pretrained(
        "/home/nvidia/Llama-3.2-3B_old/",
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
    
    return train_dataloader, args

def main(args):    
    model_split_configs, iszeroth_ls, device_to_stage_ls, stage_to_device_ls = stage_partition(args.training_methods, args.model_name_or_path)
    remote_networks = {}
    for device_idx, (model_split_config, iszeroth, device_to_stage) in enumerate(zip(model_split_configs, iszeroth_ls, device_to_stage_ls)):
        remote_networks[f"worker-{args.ip_addrs[device_idx]}"] = rpc.remote(f"worker-{args.ip_addrs[device_idx]}", DistributedSubModel, 
                                                             args=(args, device_idx, model_split_config, iszeroth, device_to_stage, device_to_stage_ls, iszeroth_ls))
    train_dataloader, args = load_train_dataloader(args)
    print("start register remote workers!!!")
    for name, rref in remote_networks.items():
        send_rref_signals = []
        for remote_name in list(remote_networks.keys()):
            if remote_name != name:
                # print("name: {}, rref: {}".format(name, rref))
                # rpc.rpc_sync(remote_name, register_rref, args=(name, rref))
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
                    remote_networks[f"worker-{args.ip_addrs[device_idx]}"].rpc_async().set_train_configuration()  #DistributedSubModel.set_train_configuration
                )
            for set_configuration_signal in set_configuration_signals:
                set_configuration_signal.wait()
            for micro_iter_idx, labels in enumerate(micro_batch_datas_labels):
                # send labels to stage[-1] and last_zerothmodel
                receive_labels_signals = []
                receive_labels_signals.append(
                    remote_networks[f"worker-{args.ip_addrs[stage_to_device_ls[-1]]}"].rpc_async().receive_labels(labels, micro_iter_idx)
                    )
                receive_labels_signals.append(
                    remote_networks[f"worker-{args.ip_addrs[iszeroth_ls.count(True)-1]}"].rpc_async().receive_labels(labels, micro_iter_idx)
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
                                remote_networks[f"worker-{args.ip_addrs[stage_to_device_ls[0]]}"].rpc_async().base_forward_only(epoch_idx, 
                                                                                                                            iter_idx, micro_iter_idx, 
                                                                                                                            micro_batch_datas_inputs[forward_mb_num])
                        else:
                            base_forward_futs[mb_num][stage_idx-1].wait()
                            base_forward_futs[mb_num][stage_idx] = \
                                remote_networks[f"worker-{args.ip_addrs[stage_to_device_ls[stage_idx]]}"].rpc_async().base_forward_only(epoch_idx, 
                                                                                                                            iter_idx, micro_iter_idx)
                for mb_num in range(args.micro_batch_num):
                    base_forward_futs[mb_num][len(stage_to_device_ls)-1].wait()
            elif args.training_methods == "only_first":   #gpipe
                S = len(stage_to_device_ls)
                M = args.micro_batch_num

                def W(stage_idx: int) -> str:
                    return f"worker-{args.ip_addrs[stage_to_device_ls[stage_idx]]}"
                F = [[None]*S for _ in range(M)]
                B = [[None]*S for _ in range(M)]
                f_started = [[False]*S for _ in range(M)]
                b_started = [[False]*S for _ in range(M)]

                while True:
                    progressed = False  # 本轮是否投递了新任务
                    # ---------- forward ----------
                    for mb in range(M):
                        for s in range(S):
                            if f_started[mb][s]:
                                continue
                            if s == 0:
                                F[mb][0] = remote_networks[W(0)].rpc_async().base_forward_only(epoch_idx, iter_idx, mb, micro_batch_datas_inputs[mb])
                                f_started[mb][0] = True
                                progressed = True
                            else:
                                prev = F[mb][s-1]
                                if prev is not None and prev.done():
                                    F[mb][s] = remote_networks[W(s)].rpc_async().base_forward_only(epoch_idx, iter_idx, mb)#, None, (W(s-1), mb))
                                    f_started[mb][s] = True
                                    progressed = True

                    # ---------- backward ----------
                    for mb in range(M):
                        if not b_started[mb][0]:
                            last_f = F[mb][S-1]
                            if last_f is not None and last_f.done():
                                B[mb][0] = remote_networks[W(S-1)].rpc_async().base_backward_only(mb)
                                b_started[mb][0] = True
                                progressed = True

                        for k in range(1, S):
                            if b_started[mb][k]:
                                continue
                            prev_bwd = B[mb][k-1]
                            if prev_bwd is not None and prev_bwd.done():
                                real_stage = S-1-k
                                B[mb][k] = remote_networks[W(real_stage)].rpc_async().base_backward_only(mb)
                                b_started[mb][k] = True
                                progressed = True

                    all_done = True
                    for mb in range(M):
                        for s in range(S):
                            if F[mb][s] is None or not F[mb][s].done():
                                all_done = False; break
                        if not all_done: break
                    if all_done:
                        for mb in range(M):
                            for k in range(S):
                                if B[mb][k] is None or not B[mb][k].done():
                                    all_done = False; break
                            if not all_done: break
                    if all_done:
                        break
                    # if F[M-1][S-1] is not None and F[M-1][S-1].done() and B[M-1][S-1] is not None and B[M-1][S-1].done():
                    #     break
                    # 避免空转占 CPU、也给 RPC 线程让出时间
                    if not progressed:
                        time.sleep(0.002)
            elif args.training_methods == "hybrid": 
                S = len(stage_to_device_ls)
                M = args.micro_batch_num
                Q = args.q_num

                nonz_stages = [i for i, z in enumerate(iszeroth_ls) if not z]
                z_stages    = [i for i, z in enumerate(iszeroth_ls) if z]
                idx_in_nonz = {s:i for i, s in enumerate(nonz_stages)}          # stage -> 反向中的“逆序索引”
                idx_in_z    = {s:i for i, s in enumerate(z_stages)}             # stage -> 估计流水线顺序

                def W(stage_idx: int) -> str:
                    return f"worker-{args.ip_addrs[ stage_to_device_ls[stage_idx] ]}"

                # ---- 这些表只放 Future；完成即代表“可驱动下游” ----
                fwd_fut   = [[None for _ in range(S)] for _ in range(M)]                 # base 前向
                bwd_fut   = [[None for _ in nonz_stages] for _ in range(M)]              # base 反向（只给非零阶段）
                est_fut   = [[[None for _ in z_stages] for _ in range(Q)] for _ in range(M)]  # 估计（M×Q×Z）

                # 记录每个 mb 在 stage0 的输入已下发
                stage0_input_sent = [False for _ in range(M)]

                def try_launch_fwd(mb: int, s: int):
                    if not (0 <= mb < M): return False
                    if fwd_fut[mb][s] is not None: return False
                    # 依赖：前一段前向完成（s==0 则无需）
                    if s > 0:
                        prev = fwd_fut[mb][s-1]
                        if (prev is None) or (not prev.done()): return False
                    if s == 0:
                        if not stage0_input_sent[mb]:
                            fut = remote_networks[W(0)].rpc_async().base_forward_only(
                                epoch_idx, iter_idx, mb, micro_batch_datas_inputs[mb]
                            )
                            stage0_input_sent[mb] = True
                        else:
                            fut = remote_networks[W(0)].rpc_async().base_forward_only(
                                epoch_idx, iter_idx, mb, None
                            )
                    else:
                        fut = remote_networks[W(s)].rpc_async().base_forward_only(
                            epoch_idx, iter_idx, mb, None#, (W(s-1), mb)
                        )
                    fwd_fut[mb][s] = fut
                    return True

                def try_launch_bwd(mb: int, s: int):
                    if s not in idx_in_nonz: return False
                    inv = len(nonz_stages) - 1 - idx_in_nonz[s]     # 反向链中的“从尾到头”索引（与你旧代码逻辑一致）
                    if bwd_fut[mb][inv] is not None: return False

                    if inv == 0:
                        last_stage = S - 1
                        req = fwd_fut[mb][last_stage]
                        if (req is None) or (not req.done()): return False
                    else:
                        prev = bwd_fut[mb][inv-1]
                        if (prev is None) or (not prev.done()): return False

                    # 发 RPC
                    fut = remote_networks[W(s)].rpc_async().base_backward_only(mb)
                    bwd_fut[mb][inv] = fut
                    return True

                def try_launch_est(mb: int, q: int, s: int):
                    if s not in idx_in_z: return False
                    zpos = idx_in_z[s]
                    if est_fut[mb][q][zpos] is not None: return False

                    # 依赖：
                    # - 第一个 zeroth 段：取输入。如果它就是 stage0，则要求 stage0 的 base_forward 已经发过第一次（输入已送达）；
                    #   否则要求“前一 zeroth 段”的同 mb/q 估计完成
                    if zpos == 0:
                        if not stage0_input_sent[mb] and fwd_fut[mb][0] is None:
                            return False
                        fut = remote_networks[W(0)].rpc_async().estimate_grads(
                            epoch_idx, iter_idx, mb, q, micro_batch_datas_inputs[mb]
                        )
                    else:
                        prev_z_stage = z_stages[zpos-1]
                        prev = est_fut[mb][q][zpos-1]
                        if (prev is None) or (not prev.done()): return False
                        fut = remote_networks[W(s)].rpc_async().estimate_grads(
                            epoch_idx, iter_idx, mb, q
                        )

                    est_fut[mb][q][zpos] = fut
                    return True

                progressed = True
                while True:
                    progressed = False

                    # 1) 前向
                    for mb in range(M):
                        for s in range(S):
                            progressed |= try_launch_fwd(mb, s)

                    # 2) 反向
                    for mb in range(M):
                        for s in reversed(range(S)):
                            progressed |= try_launch_bwd(mb, s)

                    # 3) 估计
                    for mb in range(M):
                        for q in range(Q):
                            for s in z_stages:
                                progressed |= try_launch_est(mb, q, s)

                    # 4) 结束判定放在本轮末尾；没全部完成就继续下一轮
                    all_fwd_done = all(
                        (f is not None and f.done())
                        for mb in range(M) for f in fwd_fut[mb]
                    )
                    all_bwd_done = True if len(nonz_stages) == 0 else all(
                        (f is not None and f.done())
                        for mb in range(M) for f in bwd_fut[mb]
                    )
                    all_est_done = True if len(z_stages) == 0 else all(
                        (f is not None and f.done())
                        for mb in range(M) for q in range(Q) for f in est_fut[mb][q]
                    )

                    if all_fwd_done and all_bwd_done and all_est_done:
                        break

                    if not progressed:
                        time.sleep(0.001)  # 给远端一点时间推进
                # print("fwd_fut: ", fwd_fut)
                # print("bwd_fut: ", bwd_fut)
                # print("est_fut: ", est_fut)
                for mb in range(M):
                    for s in range(S):
                        if fwd_fut[mb][s] is not None:
                            fwd_fut[mb][s].wait()
                    for inv in range(len(nonz_stages)):
                        if bwd_fut[mb][inv] is not None:
                            bwd_fut[mb][inv].wait()
                    for q in range(Q):
                        for zpos in range(len(z_stages)):
                            if est_fut[mb][q][zpos] is not None:
                                est_fut[mb][q][zpos].wait()

                # 如果最后一段在你那边需要同步 loss 给零阶段，这里保持你原逻辑
                remote_networks[ W(S-1) ].rpc_sync().send_base_loss()
            elif args.training_methods == "hybrid_v1": 
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
                                # print("fowarad satge:{}, micro-batch {}".format(stage_idx, forward_mb_num))
                                if stage_idx == 0:
                                    base_forward_futs[forward_mb_num][stage_idx] = \
                                        remote_networks[f"worker-{args.ip_addrs[stage_to_device_ls[0]]}"].rpc_async().base_forward_only(epoch_idx,
                                                                                                                                    iter_idx, forward_mb_num,
                                                                                                                                    micro_batch_datas_inputs[forward_mb_num])
                                else:
                                    base_forward_futs[forward_mb_num][stage_idx-1].wait()
                                    base_forward_futs[forward_mb_num][stage_idx] = \
                                        remote_networks[f"worker-{args.ip_addrs[stage_to_device_ls[stage_idx]]}"].rpc_async().base_forward_only(epoch_idx, iter_idx, forward_mb_num)
                    else:
                        if step_idx <= max(args.micro_batch_num, len(stage_to_device_ls)):
                            for stage_idx, iszeroth in enumerate(iszeroth_ls):
                                if not iszeroth:
                                    break
                                forward_mb_num = step_idx - stage_idx
                                if 0 <= forward_mb_num < args.micro_batch_num and iszeroth_ls[stage_idx]:
                                    # print("fowarad satge:{}, micro-batch {}".format(stage_idx, forward_mb_num))
                                    if stage_idx == 0:
                                        base_forward_futs[forward_mb_num][stage_idx] = \
                                            remote_networks[f"worker-{args.ip_addrs[stage_to_device_ls[0]]}"].rpc_async().base_forward_only(epoch_idx,
                                                                                                                                        iter_idx, forward_mb_num,
                                                                                                                                        micro_batch_datas_inputs[forward_mb_num])
                                    else:
                                        base_forward_futs[forward_mb_num][stage_idx-1].wait()
                                        base_forward_futs[forward_mb_num][stage_idx] = \
                                            remote_networks[f"worker-{args.ip_addrs[stage_to_device_ls[stage_idx]]}"].rpc_async().base_forward_only(epoch_idx, iter_idx, forward_mb_num)
                        
                        for stage_idx, device_idx in enumerate(stage_to_device_ls):
                            if not iszeroth_ls[stage_idx]:
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
                                            remote_networks[f"worker-{args.ip_addrs[stage_to_device_ls[0]]}"].rpc_async().base_forward_only(epoch_idx,
                                                                                                                                        iter_idx, forward_mb_num,
                                                                                                                                        micro_batch_datas_inputs[forward_mb_num])
                                    else:
                                        base_forward_futs[forward_mb_num][stage_idx-1].wait()
                                        base_forward_futs[forward_mb_num][stage_idx] = \
                                            remote_networks[f"worker-{args.ip_addrs[stage_to_device_ls[stage_idx]]}"].rpc_async().base_forward_only(epoch_idx, iter_idx, forward_mb_num)
                                elif (backward_mb_num is not None) and (0 <= backward_mb_num < args.micro_batch_num):
                                    # print("backward satge:{}, micro-batch {}".format(stage_idx, backward_mb_num))
                                    if inverse_stage == 0:
                                        base_forward_futs[backward_mb_num][len(stage_to_device_ls)-1].wait()
                                        base_backward_futs[backward_mb_num][inverse_stage] = \
                                            remote_networks[f"worker-{args.ip_addrs[stage_to_device_ls[stage_idx]]}"].rpc_async().base_backward_only(backward_mb_num)
                                    else:
                                        base_backward_futs[backward_mb_num][inverse_stage-1].wait()
                                        base_backward_futs[backward_mb_num][inverse_stage] = \
                                            remote_networks[f"worker-{args.ip_addrs[stage_to_device_ls[stage_idx]]}"].rpc_async().base_backward_only(backward_mb_num)
                    #estimate
                    for stage_idx, iszeroth in enumerate(iszeroth_ls):
                        if not iszeroth:
                            break
                        estimate_mb_num = (step_idx - stage_idx - args.micro_batch_num) // args.q_num
                        estimate_q_cnt = (step_idx - stage_idx - args.micro_batch_num) % args.q_num
                        if (0 <= estimate_mb_num < args.micro_batch_num):
                            # print("forward satge:{}, micro-batch {} estimate_q_cnt {}".format(stage_idx, estimate_mb_num, estimate_q_cnt))
                            if stage_idx == 0:
                                if estimate_q_cnt == 0:
                                    estimate_forward_futs[estimate_mb_num][estimate_q_cnt][stage_idx] = \
                                        remote_networks[f"worker-{args.ip_addrs[stage_to_device_ls[stage_idx]]}"].rpc_async().estimate_grads(epoch_idx, 
                                                                                                                                         iter_idx, estimate_mb_num, estimate_q_cnt,
                                                                                                                                         micro_batch_datas_inputs[estimate_mb_num])
                                else:
                                    estimate_forward_futs[estimate_mb_num][estimate_q_cnt][stage_idx] = \
                                        remote_networks[f"worker-{args.ip_addrs[stage_to_device_ls[stage_idx]]}"].rpc_async().estimate_grads(epoch_idx, 
                                                                                                                                iter_idx, estimate_mb_num, estimate_q_cnt)
                            else:
                                estimate_forward_futs[estimate_mb_num][estimate_q_cnt][stage_idx-1].wait()
                                estimate_forward_futs[estimate_mb_num][estimate_q_cnt][stage_idx] = \
                                    remote_networks[f"worker-{args.ip_addrs[stage_to_device_ls[stage_idx]]}"].rpc_async().estimate_grads(epoch_idx, 
                                                                                                                             iter_idx, estimate_mb_num, estimate_q_cnt)
                    # print("-"*50)
                for mb_num in range(args.micro_batch_num):
                    base_backward_futs[mb_num][iszeroth_ls.count(False)-1].wait()
                for mb_num in range(args.micro_batch_num):
                    for q_cnt in range(args.q_num):
                        estimate_forward_futs[mb_num][q_cnt][iszeroth_ls.count(True)-1].wait()

                remote_networks[f"worker-{args.ip_addrs[stage_to_device_ls[-1]]}"].rpc_sync().send_base_loss()
            
            end = time.time()
            print("Iter: {}, time: {}".format(iter_idx, end-start))
            #optimizer
            optimizer_signals = []
            for stage_idx, device_idx in enumerate(stage_to_device_ls):
                optimizer_signals.append(
                    remote_networks[f"worker-{args.ip_addrs[device_idx]}"].rpc_async().network_optimzier()
                )
            for optimizer_signal in optimizer_signals:
                optimizer_signal.wait()

            end = time.time()
            print("Iter with optimizer: {}, time: {}".format(iter_idx, end-start))

def init_process(rank, world_size, args):
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    os.environ["GLOO_SOCKET_IFNAME"] = args.ifname
    os.environ["TP_SOCKET_IFNAME"] = args.ifname
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    
    opts = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=32,
        rpc_timeout=300
    )
    if rank == 0:
        rpc.init_rpc(f"master", rank=rank, world_size=world_size, rpc_backend_options=opts)
        print("master comm success!!")
        main(args)
    else:
        final_ip_addr = args.ip_addr.split('.')[-1]
        rpc.init_rpc(f"worker-{final_ip_addr}", rank=rank, world_size=world_size, rpc_backend_options=opts)
        print("rank {} comm success!!".format(rank))
    rpc.shutdown()

if __name__ == "__main__":
    args = parse_args()
    args.master_addr = "192.168.1.125"
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    set_all_seeds(args.seed)
    args.ip_addr = get_ip_address(args.ifname)
    # device = f"cuda:{args.device}"
    args.ip_addrs = args.ip_addrs.split(',')
    world_size = 1 + len(args.ip_addrs)
    final_ip_addr = args.ip_addr.split('.')[-1]
    
    if args.ip_addr == args.master_addr:
        print("master ip_addr:", args.ip_addr)
        # mp.spawn(init_process, args=(world_size, args), nprocs=2, join=True)
        init_process(0, world_size, args)
    else:
        print("ip_addr:", args.ip_addr)
        rank = args.ip_addrs.index(final_ip_addr) + 1
        # final_ip_addr = args.ip_addr.split('.')[-1]
        # rank = args.ip_addrs.index(final_ip_addr)
        init_process(rank, world_size, args)
