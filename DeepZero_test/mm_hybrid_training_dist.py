import os
import io
import time
import math
import socket
import fcntl
import struct
import random
import argparse
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, set_seed

from parse_args import parse_args
from data.math_dataloader import make_supervised_data_module, smart_tokenizer_and_embedding_resize
from loss_utils import ForCausalLMLoss
from llm_profile import stage_partition
from distributed_model_dist import DistributedSubModel
from datetime import timedelta

import io
import math
import torch
import torch.distributed as dist
from collections import deque
from datasets import load_dataset
from transformers.data import DataCollatorWithPadding
try:
    import torch_npu
    HAS_TORCH_NPU = True
except ModuleNotFoundError:
    HAS_TORCH_NPU = False


def get_tensor_mem(obj):
    return obj.element_size() * obj.nelement()

def get_mem(obj, visited=None):
    if visited is None:
        visited = set()

    if torch.is_tensor(obj):
        if id(obj) in visited:  # 避免重复统计
            return 0
        visited.add(id(obj))
        return get_tensor_mem(obj)

    elif isinstance(obj, dict):
        return sum(get_mem(v, visited) for v in obj.values())

    elif isinstance(obj, (list, tuple)):
        return sum(get_mem(v, visited) for v in obj)

    else:
        return 0
    
_OUTSTANDING_SENDS = []

def send_obj(obj, dst: int, tag: int):
    buf = io.BytesIO()
    torch.save(obj, buf, _use_new_zipfile_serialization=True)
    raw = buf.getbuffer()  # 不复制
    ln = torch.tensor([len(raw)], dtype=torch.int64)
    w1 = dist.isend(ln, dst=dst, tag=tag)
    payload = torch.frombuffer(bytearray(raw), dtype=torch.uint8)  # 可写，无警告
    w2 = dist.isend(payload, dst=dst, tag=tag + 1)
    _OUTSTANDING_SENDS.extend([("hdr", dst, tag, w1, ln),
                               ("pay", dst, tag+1, w2, payload)])

def recv_obj(src: int, tag: int, map_location="cpu"):
    ln = torch.empty(1, dtype=torch.int64)
    dist.recv(ln, src=src, tag=tag)
    n = int(ln.item())
    payload = torch.empty(n, dtype=torch.uint8)
    dist.recv(payload, src=src, tag=tag + 1)
    # 这里不要 tolist()
    buf = io.BytesIO(payload.numpy().tobytes())
    return torch.load(buf, map_location=map_location)

def flush_sends():
    global _OUTSTANDING_SENDS
    for kind, dst, tag, work, tensor_ref in _OUTSTANDING_SENDS:
        try:
            work.wait()
        except Exception as e:
            print(f"[rank {dist.get_rank()}] isend(kind={kind}, dst={dst}, tag={tag}) wait failed: {e}", flush=True)
            raise
    _OUTSTANDING_SENDS = []

TAG_BASE_FWD = 10_000
TAG_BASE_BWD = 20_000
TAG_EST      = 30_000
TAG_LBL_LAST = 40_000
TAG_LBL_Z    = 50_000
TAG_LOSS     = 60_000
def tag_fwd(mb): return TAG_BASE_FWD + mb
def tag_bwd(mb): return TAG_BASE_BWD + mb
def tag_est(mb, q): return TAG_EST + (mb * 1000) + q
def tag_lbl_last(mb): return TAG_LBL_LAST + mb
def tag_lbl_z(mb):    return TAG_LBL_Z + mb
def tag_loss():       return TAG_LOSS


DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

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

def load_train_dataloader(args):
    if args.dataset == "agnews":
        dataset_path = "./ag_news"
        id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
        label2id = {v: k for k, v in id2label.items()}
        ds = load_dataset(dataset_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        def preprocess_function(examples):
            tokenized = tokenizer(examples["text"], max_length=args.max_seq_len, truncation=True)
            tokenized["labels"] = examples["label"]
            return tokenized
        tokenized_ds = ds.map(preprocess_function, batched=True, remove_columns=["text"])
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        train_loader = DataLoader(
            tokenized_ds["train"], batch_size=args.train_batch_size, shuffle=True, collate_fn=data_collator
        )
        # eval_loader = DataLoader(
        #     tokenized_ds["test"], batch_size=args.eval_batch_size, shuffle=False, collate_fn=data_collator
        # )
        
        args.vocab_size = len(tokenizer)
        args.micro_batch_num = int(args.train_batch_size // args.micro_batch_size)
        args.total_iterations = len(train_loader)
        # print(f"Training set size: {len(tokenized_ds['train'])} samples")
        # print(f"Test set size: {len(tokenized_ds["test"])} samples")
        return train_loader, args
    elif args.dataset == "multirc":
        pass
    elif args.dataset == "squad":
        pass
    elif args.dataset == "math":
        tokenizer = AutoTokenizer.from_pretrained(
            "../Llama-3.2-3B_old/",
            model_max_length=args.max_seq_len,
            padding_side="right",
            use_fast=False,
        )
        special_tokens_dict = {}
        if tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

        # 仅 rank0 需要真正加载模型做 resize，其它 rank 不用
        if dist.get_rank() == 0:
            model_config = AutoConfig.from_pretrained("../Llama-3.2-3B_old/")
            tmp_model = AutoModelForCausalLM.from_pretrained(
                "../Llama-3.2-3B_old/",
                config=model_config,
                ignore_mismatched_sizes=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=special_tokens_dict,
                tokenizer=tokenizer,
                model=tmp_model,
            )
            del tmp_model
            torch.cuda.empty_cache()

        args.vocab_size = len(tokenizer)
        args.micro_batch_num = int(args.batch_size // args.micro_batch_size)
        train_dataset, data_collator, train_dataloader = make_supervised_data_module(tokenizer, args)
        args.total_iterations = len(train_dataloader)
        return train_dataloader, args

def run_training(args):
    rank = dist.get_rank()
    world = dist.get_world_size()

    # 1) 分段（所有 rank 都跑一遍，得到一致结果）
    model_split_configs, iszeroth_ls, device_to_stage_ls, stage_to_device_ls, to_device_ls = stage_partition(
        args.training_methods, args.model_name_or_path
    )
    assert world == len(stage_to_device_ls), f"world_size={world} 必须等于 stage 数={len(stage_to_device_ls)}"

    S = world
    s = rank
    to_device = to_device_ls[s]
    # if to_device == "npu":
    #     import torch_npu
        
    if s == 0:
        train_dataloader, args = load_train_dataloader(args)  # 这里会设置 args.total_iterations / args.micro_batch_num / args.vocab_size
        meta = {
            "total_iterations": args.total_iterations,
            "micro_batch_num": args.micro_batch_num,
            "vocab_size": args.vocab_size,
            "max_seq_len": args.max_seq_len,
        }
    else:
        train_dataloader = None
        meta = None

    obj = [meta]
    dist.broadcast_object_list(obj, src=0)
    meta = obj[0]
    # 在非 rank0 上补全 args
    if s != 0:
        args.total_iterations = meta["total_iterations"]
        args.micro_batch_num  = meta["micro_batch_num"]
        args.vocab_size       = meta["vocab_size"]
        args.max_seq_len      = meta["max_seq_len"]

    M = args.micro_batch_num
    Q = args.q_num

    z_stages   = [i for i, z in enumerate(iszeroth_ls) if z]
    nonz_stages= [i for i, z in enumerate(iszeroth_ls) if not z]
    z_count    = len(z_stages)
    first_nonz = z_count
    last_stage = S - 1
    last_zeroth= z_count - 1 if z_count > 0 else None

    sub = DistributedSubModel(args, model_split_configs[s], iszeroth_ls, device=to_device, stage_idx=s, iszeroth=iszeroth_ls[s], total_stages=S)
    data_iter = iter(train_dataloader) if s == 0 else None

    for epoch_idx in range(args.epochs):
        if s == 0:
            print(f"[epoch {epoch_idx}] iters={args.total_iterations}", flush=True)

        for iter_idx in range(args.total_iterations):
            sub.set_train_configuration()
            if s == 0:  #comm inputs and labels
                base_inputs = next(data_iter)
                # print(base_inputs)
                batch_inputs = {k:v for k,v in base_inputs.items() if k != "labels"}
                batch_labels = base_inputs["labels"]
                # batch_inputs, batch_labels = next(data_iter)
                
                micro_input_ids_ls = torch.split(batch_inputs["input_ids"], args.micro_batch_size, dim=0)
                micro_attention_mask_ls = torch.split(batch_inputs["attention_mask"], args.micro_batch_size, dim=0)
                micro_inputs = [
                    dict(input_ids=ids, attention_mask=attn)
                    for ids, attn in zip(micro_input_ids_ls, micro_attention_mask_ls)
                ]
                micro_labels = torch.split(batch_labels, args.micro_batch_size, dim=0)

                for mb in range(M):
                    send_obj(micro_labels[mb], dst=last_stage, tag=tag_lbl_last(mb))
                    if last_zeroth is not None:
                        send_obj(micro_labels[mb], dst=last_zeroth, tag=tag_lbl_z(mb))
            else:
                micro_inputs = None
                if s == last_stage:
                    sub.labels = {mb: recv_obj(src=0, tag=tag_lbl_last(mb)) for mb in range(M)}
                if last_zeroth is not None and s == last_zeroth:
                    sub.labels = {mb: recv_obj(src=0, tag=tag_lbl_z(mb)) for mb in range(M)}
            flush_sends()
            dist.barrier()
            if s == 0:
                # print(f"iter {iter_idx} start!!")
                start = time.time()
            
            if args.training_methods == "only_first":  #dapple
                base_forward_total_steps = len(stage_to_device_ls) + args.micro_batch_num - 1
                base_backward_total_steps = len(stage_to_device_ls) + args.micro_batch_num - 1
                total_steps = 2 * (len(stage_to_device_ls) + args.micro_batch_num - 1)
                for step_idx in range(total_steps):
                    # print(f"step {step_idx} start!!")
                    if step_idx < base_forward_total_steps:
                        forward_mb_num = step_idx - s
                        if 0 <= forward_mb_num < args.micro_batch_num:
                            # print("stage: {}, forward mb: {} done!!".format(s, forward_mb_num))
                            if s == 0:
                                out = sub.base_forward_only(epoch_idx, iter_idx, forward_mb_num, inputs=micro_inputs[forward_mb_num])
                                # print("out: ", out)
                                # mem_bytes = get_mem(out)
                                # print(f"forward {mem_bytes / 1024**2:.4f} MB")
                                if s < last_stage:
                                    send_obj(out, dst=s+1, tag=tag_fwd(forward_mb_num))
                            else:
                                inp = recv_obj(src=s-1, tag=tag_fwd(forward_mb_num))
                                # print("inp: ", inp)
                                # import sys
                                # sys.exit(0)
                                out = sub.base_forward_only(epoch_idx, iter_idx, forward_mb_num, inputs=inp)
                                if s < last_stage:
                                    send_obj(out, dst=s+1, tag=tag_fwd(forward_mb_num))
                    else:
                        inverse_stage = len(stage_to_device_ls) - 1 - s
                        backward_mb_num = step_idx - base_forward_total_steps - inverse_stage
                        if 0 <= backward_mb_num < args.micro_batch_num:
                            # print("stage: {}, backward mb: {}".format(s, backward_mb_num))
                            if inverse_stage == 0:
                                grad = sub.base_backward_only(backward_mb_num)
                                # mem_bytes = get_mem(grad)
                                # print(f"grad {mem_bytes / 1024**2:.4f} MB")
                                if s > 0:
                                    send_obj(grad, dst=s-1, tag=tag_bwd(backward_mb_num))
                            else:
                                grad_in = recv_obj(src=s+1, tag=tag_bwd(backward_mb_num))
                                grad = sub.base_backward_only(backward_mb_num, grad_in)
                                if s > 0:
                                    send_obj(grad, dst=s-1, tag=tag_bwd(backward_mb_num))
                    # print("-"*50)
            elif args.training_methods == "pipedream":
                base_forward_total_steps = len(stage_to_device_ls) + args.micro_batch_num - 1
                base_backward_total_steps = len(stage_to_device_ls) + args.micro_batch_num - 1
                total_steps = 2 * (len(stage_to_device_ls) + args.micro_batch_num - 1)
                for step_idx in range(total_steps):
                    # print(f"step {step_idx} start!!")
                    if step_idx < len(stage_to_device_ls):
                        forward_mb_num = step_idx - s
                        if 0 <= forward_mb_num < args.micro_batch_num:
                            # print("stage: {}, forward mb: {}".format(s, forward_mb_num))
                            if s == 0:
                                out = sub.base_forward_only(epoch_idx, iter_idx, forward_mb_num, inputs=micro_inputs[forward_mb_num])
                                if s < last_stage:
                                    send_obj(out, dst=s+1, tag=tag_fwd(forward_mb_num))
                            else:
                                inp = recv_obj(src=s-1, tag=tag_fwd(forward_mb_num))
                                out = sub.base_forward_only(epoch_idx, iter_idx, forward_mb_num, inputs=inp)
                                if s < last_stage:
                                    send_obj(out, dst=s+1, tag=tag_fwd(forward_mb_num))
                    else:
                        forward_mb_num, backward_mb_num = None, None
                        inverse_stage = len(stage_to_device_ls) - 1 - s
                        if ((step_idx - s) % 2) == 0:
                            forward_mb_num = (step_idx - s) // 2
                        else:
                            backward_mb_num = ((step_idx - (len(stage_to_device_ls)-1)) - inverse_stage) // 2
                        
                        if (forward_mb_num is not None) and (inverse_stage < forward_mb_num < args.micro_batch_num):
                            # print("stage: {}, forward mb: {}".format(s, forward_mb_num))
                            if s == 0:
                                out = sub.base_forward_only(epoch_idx, iter_idx, forward_mb_num, inputs=micro_inputs[forward_mb_num])
                                if s < last_stage:
                                    send_obj(out, dst=s+1, tag=tag_fwd(forward_mb_num))
                            else:
                                inp = recv_obj(src=s-1, tag=tag_fwd(forward_mb_num))
                                out = sub.base_forward_only(epoch_idx, iter_idx, forward_mb_num, inputs=inp)
                                if s < last_stage:
                                    send_obj(out, dst=s+1, tag=tag_fwd(forward_mb_num))
                        elif (backward_mb_num is not None) and (0 <= backward_mb_num < args.micro_batch_num):
                            # print("stage: {}, backward mb: {}".format(s, backward_mb_num))
                            if inverse_stage == 0:
                                grad = sub.base_backward_only(backward_mb_num)
                                if s > 0:
                                    send_obj(grad, dst=s-1, tag=tag_bwd(backward_mb_num))
                            else:
                                grad_in = recv_obj(src=s+1, tag=tag_bwd(backward_mb_num))
                                # print("stage: {}, backward mb: {}".format(s, bmb))
                                grad = sub.base_backward_only(backward_mb_num, grad_in)
                                if s > first_nonz:
                                    send_obj(grad, dst=s-1, tag=tag_bwd(backward_mb_num))
                    print("-"*50)
            
                base_forward_total_steps = len(set(stage_to_device_ls)) + args.micro_batch_num - 1
                base_backward_total_steps = len(set(stage_to_device_ls)) + args.micro_batch_num - 1
                total_steps = 2 * (len(set(stage_to_device_ls)) + args.micro_batch_num - 1)
                for step_idx in range(total_steps):
                    # print(f"step {step_idx} start!!")
                    if step_idx < base_forward_total_steps:
                        forward_mb_num = step_idx - s
                        if 0 <= forward_mb_num < args.micro_batch_num:
                            # print("stage: {}, forward mb: {} done!!".format(s, forward_mb_num))
                            if s == 0:
                                out = sub.base_forward_only(epoch_idx, iter_idx, forward_mb_num, inputs=micro_inputs[forward_mb_num])
                                # print("out: ", out)
                                # mem_bytes = get_mem(out)
                                # print(f"forward {mem_bytes / 1024**2:.4f} MB")
                                if s < last_stage:
                                    send_obj(out, dst=s+1, tag=tag_fwd(forward_mb_num))
                            else:
                                inp = recv_obj(src=s-1, tag=tag_fwd(forward_mb_num))
                                # print("inp: ", inp)
                                # import sys
                                # sys.exit(0)
                                out = sub.base_forward_only(epoch_idx, iter_idx, forward_mb_num, inputs=inp)
                                if s < last_stage:
                                    send_obj(out, dst=s+1, tag=tag_fwd(forward_mb_num))
                    else:
                        inverse_stage = len(stage_to_device_ls) - 1 - s
                        backward_mb_num = step_idx - base_forward_total_steps - inverse_stage
                        if 0 <= backward_mb_num < args.micro_batch_num:
                            # print("stage: {}, backward mb: {}".format(s, backward_mb_num))
                            if inverse_stage == 0:
                                grad = sub.base_backward_only(backward_mb_num)
                                # mem_bytes = get_mem(grad)
                                # print(f"grad {mem_bytes / 1024**2:.4f} MB")
                                if s > 0:
                                    send_obj(grad, dst=s-1, tag=tag_bwd(backward_mb_num))
                            else:
                                grad_in = recv_obj(src=s+1, tag=tag_bwd(backward_mb_num))
                                grad = sub.base_backward_only(backward_mb_num, grad_in)
                                if s > 0:
                                    send_obj(grad, dst=s-1, tag=tag_bwd(backward_mb_num))
                pass
            elif args.training_methods == "mixpipe":
                pass
            elif args.training_methods == "zero_bubble":
                pass
            elif args.training_methods == "hybrid":
                estimate_total_steps = max(M + Q*M + z_count - 1, 2*(S + M - 1))
                for step_idx in range(estimate_total_steps):
                    if step_idx < S:
                        mb = step_idx - s
                        if 0 <= mb < M:
                            if s == 0:
                                out = sub.base_forward_only(epoch_idx, iter_idx, mb, inputs=micro_inputs[mb])
                                # print("stage: {}, forward mb: {} done!!".format(s, mb))
                                if s < last_stage:
                                    send_obj(out, dst=s+1, tag=tag_fwd(mb))
                            else:
                                inp = recv_obj(src=s-1, tag=tag_fwd(mb))
                                # print("stage: {}, forward mb: {}".format(s, mb))
                                out = sub.base_forward_only(epoch_idx, iter_idx, mb, inputs=inp)
                                if s < last_stage:
                                    send_obj(out, dst=s+1, tag=tag_fwd(mb))
                    else:
                        if step_idx <= max(M, S):
                            mb = step_idx - s
                            if 0 <= mb < M and s in z_stages:
                                if s == 0:
                                    out = sub.base_forward_only(epoch_idx, iter_idx, mb, inputs=micro_inputs[mb])
                                    if s < last_stage:
                                        send_obj(out, dst=s+1, tag=tag_fwd(mb))
                                else:
                                    inp = recv_obj(src=s-1, tag=tag_fwd(mb))
                                    # print("stage: {}, forward mb: {}".format(s, mb))
                                    out = sub.base_forward_only(epoch_idx, iter_idx, mb, inputs=inp)
                                    if s < last_stage:
                                        send_obj(out, dst=s+1, tag=tag_fwd(mb))
                        if s in nonz_stages:
                            inverse_stage = (S - 1) - s
                            fmb = None
                            bmb = None
                            if ((step_idx - s) % 2) == 0:
                                fmb = (step_idx - s) // 2
                            else:
                                bmb = ((step_idx - (S - 1)) - inverse_stage) // 2

                            if (bmb is not None) and (0 <= bmb < M):
                                # backward：尾段先自己 backward，否则收来自 s+1
                                if s == last_stage:
                                    # print("stage: {}, backward mb: {}".format(s, bmb))
                                    grad = sub.base_backward_only(bmb)  # 返回传给前一段的梯度
                                    if s > 0:
                                        send_obj(grad, dst=s-1, tag=tag_bwd(bmb))
                                else:
                                    grad_in = recv_obj(src=s+1, tag=tag_bwd(bmb))
                                    # print("stage: {}, backward mb: {}".format(s, bmb))
                                    grad = sub.base_backward_only(bmb, grad_in)
                                    if s > first_nonz:
                                        send_obj(grad, dst=s-1, tag=tag_bwd(bmb))
                            elif (fmb is not None) and (inverse_stage < fmb < M):
                                if s == 0:
                                    out = sub.base_forward_only(epoch_idx, iter_idx, mb, inputs=micro_inputs[mb])
                                    # print("stage: {}, forward mb: {} done!!".format(s, fmb))
                                    if s < last_stage:
                                        send_obj(out, dst=s+1, tag=tag_fwd(mb))
                                else:
                                    inp = recv_obj(src=s-1, tag=tag_fwd(fmb))
                                    # print("stage: {}, forward mb: {}".format(s, fmb))
                                    out = sub.base_forward_only(epoch_idx, iter_idx, fmb, inputs=inp)
                                    if s < last_stage:
                                        send_obj(out, dst=s+1, tag=tag_fwd(fmb))

                    # C) estimate（整个流水线，从 zeroth 前缀开始一直到最后段）
                    if s in z_stages:
                        est_mb = (step_idx - s - M) // Q
                        est_q  = (step_idx - s - M) %  Q
                        if 0 <= est_mb < M:
                            if s == 0:
                                if est_q == 0:
                                    # print("stage: {}, estimate_grads mb: {} est_q: {}".format(s, est_mb, est_q))
                                    est_out = sub.estimate_grads(epoch_idx, iter_idx, est_mb, est_q, inputs=micro_inputs[est_mb])
                                else:
                                    # print("stage: {}, estimate_grads mb: {} est_q: {}".format(s, est_mb, est_q))
                                    est_out = sub.estimate_grads(epoch_idx, iter_idx, est_mb, est_q, inputs=None)
                                # 往下游传
                                if s < last_stage:
                                    send_obj(est_out, dst=s+1, tag=tag_est(est_mb, est_q))
                            elif sub.last_zeroth_model:
                                # 收来自 s-1
                                est_in = recv_obj(src=s-1, tag=tag_est(est_mb, est_q))
                                # print("stage: {}, estimate_grads mb: {} est_q: {}".format(s, est_mb, est_q))
                                est_out = sub.estimate_grads(epoch_idx, iter_idx, est_mb, est_q, inputs=est_in)
                                # if s < last_stage:
                                #     send_obj(est_out, dst=s+1, tag=tag_est(est_mb, est_q))
                            else:
                                # 收来自 s-1
                                est_in = recv_obj(src=s-1, tag=tag_est(est_mb, est_q))
                                # print("stage: {}, estimate_grads mb: {} est_q: {}".format(s, est_mb, est_q))
                                est_out = sub.estimate_grads(epoch_idx, iter_idx, est_mb, est_q, inputs=est_in)
                                if s < last_stage:
                                    send_obj(est_out, dst=s+1, tag=tag_est(est_mb, est_q))
                    # elif s > z_stages[-1] if z_stages else False:
                    #     # 非 zeroth 段也要承接估计流水线
                    #     # 需要判断该 step 是否正好有 (mb,q) 从上游来
                    #     est_mb = (step_idx - (z_stages[-1]) - M) // Q if z_stages else None
                    #     # 更简单可靠：尝试根据 (step_idx, s) 推出可能的 est_mb/est_q
                    #     est_mb = (step_idx - s - M + (s - z_stages[-1])) // Q if z_stages else None
                    #     est_q  = (step_idx - s - M + (s - z_stages[-1])) %  Q if z_stages else None
                    #     if z_stages and 0 <= est_mb < M:
                    #         est_in = recv_obj(src=s-1, tag=tag_est(est_mb, est_q))
                    #         print("stage: {}, estimate_grads mb: {} est_q: {}".format(s, mb, est_q))
                    #         est_out = sub.estimate_grads(epoch_idx, iter_idx, est_mb, est_q, inputs=est_in)
                    #         if s < last_stage:
                    #             send_obj(est_out, dst=s+1, tag=tag_est(est_mb, est_q))
                    #         else:
                    #             # 最后一段计算估计损失并仅在本地缓存（sub 内部已处理）
                    #             pass
                
                if s == last_stage:
                    send_obj(sub.base_loss_eval, dst=last_zeroth if last_zeroth is not None else 0, tag=tag_loss())
                elif last_zeroth is not None and s == last_zeroth:
                    base_loss_eval = recv_obj(src=last_stage, tag=tag_loss())
                    deriv = sub.zeroth_compute_directional_derivative(base_loss_eval)  # dict[mb][q] -> scalar
                    for z in z_stages:
                        if z == last_zeroth:
                            continue
                        send_obj(deriv, dst=z, tag=tag_loss()+1)
                if z_count > 0 and (s in z_stages) and (s != last_zeroth):
                    deriv = recv_obj(src=last_zeroth, tag=tag_loss()+1)
                    sub.update_zeroth_model(deriv)
                    
            flush_sends()   # 把当前迭代的所有未完成 isend 冲干净
            dist.barrier()
            # 优化器 step
            sub.network_optimizer()

            if s == 0:
                end = time.time()
                print(f"[iter {iter_idx}] done time: {end-start}.")

def _infer_rank_and_world(args):
    local_ip = get_ip_address(args.ifname)
    last_octet = local_ip.split(".")[-1]

    # 归一化 ip_addrs
    if isinstance(args.ip_addrs, str):
        args.ip_addrs = [x for x in args.ip_addrs.split(",") if x.strip()]

    # 单进程自检（不分布式）
    if not args.ip_addrs:
        return 0, 1, local_ip

    participants = args.ip_addrs  # ★ 现在包含所有 stage（含 stage0）
    if last_octet not in participants:
        raise RuntimeError(
            f"[DistInit] 本机 {local_ip} 的末段 {last_octet} 不在 --ip_addrs={participants} 中。\n"
            f"请把本机加入 --ip_addrs，并保持顺序即为 stage 顺序。"
        )

    world_size = len(participants)  
    rank = participants.index(last_octet)

    assert args.master_addr.split(".")[-1] == participants[0], \
        f"master_addr 需指向 stage0 主机；当前 master={args.master_addr}，但 ip_addrs[0]={participants[0]}"

    return rank, world_size, local_ip


def _print_dist_banner(where, **kw):
    pad = max(len(k) for k in kw.keys())
    lines = [f"===== {where} ====="]
    for k, v in kw.items():
        lines.append(f"{k.rjust(pad)} : {v}")
    print("\n".join(lines), flush=True)


def main():
    args = parse_args()
    set_all_seeds(args.seed)

    if isinstance(args.ip_addrs, str):
        args.ip_addrs = [x for x in args.ip_addrs.split(",") if x.strip()]
    rank, world_size, local_ip = _infer_rank_and_world(args)
    os.environ["GLOO_SOCKET_IFNAME"] = args.ifname

    _print_dist_banner(
        "Dist Bootstrap (before init)",
        local_ip=local_ip,
        master_addr=args.master_addr,
        master_port=args.master_port,
        ifname=args.ifname,
        rank=rank,
        world_size=world_size,
        ip_addrs=args.ip_addrs,
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
    )

    # 关键一致性检查（提前报错，不要卡死）
    if world_size > 1:
        if rank == 0 and local_ip != args.master_addr:
            raise RuntimeError(
                f"[DistInit] 你被推断为 rank0，但本机 IP={local_ip} != master_addr={args.master_addr}。\n"
                f"请把 --master_addr 改成本机 IP，或把 args.ip_addrs 配置正确。"
            )
        if rank != 0 and local_ip == args.master_addr:
            raise RuntimeError(
                f"[DistInit] 你被推断为非 0 rank，但本机 IP 却等于 master_addr（{local_ip}）。"
            )

    # 初始化进程组（会在这里等待所有 world_size 个进程都加入）
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://{args.master_addr}:{args.master_port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=120),
    )

    _print_dist_banner(
        "Dist Bootstrap (after init)",
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
    )

    # try:
    run_training(args)
    # finally:
    # 为了更容易定位谁没走到这里，我们给每个 rank 打印一下
    print(f"[rank {dist.get_rank()}] training done, entering barrier...", flush=True)
    dist.barrier()
    print(f"[rank {dist.get_rank()}] destroying process group", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
