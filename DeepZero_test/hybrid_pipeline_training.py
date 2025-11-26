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

# def send_obj(obj, dst: int, tag: int):
#     buf = io.BytesIO()
#     torch.save(obj, buf)
#     raw = buf.getvalue()

#     ln = torch.tensor([len(raw)], dtype=torch.int64)
#     w1 = dist.isend(ln, dst=dst, tag=tag)  # header
#     # 注意：必须把 ByteTensor 对象保活到 wait 完毕
#     payload = torch.frombuffer(memoryview(raw), dtype=torch.uint8).clone()
#     w2 = dist.isend(payload, dst=dst, tag=tag + 1)  # body

#     # 保活（顺序也保留，便于排查）
#     _OUTSTANDING_SENDS.append(("hdr", dst, tag, w1, ln))
#     _OUTSTANDING_SENDS.append(("pay", dst, tag + 1, w2, payload))

# def recv_obj(src: int, tag: int):
#     ln = torch.empty(1, dtype=torch.int64)
#     dist.recv(ln, src=src, tag=tag)
#     n = int(ln.item())
#     payload = torch.empty(n, dtype=torch.uint8)
#     dist.recv(payload, src=src, tag=tag + 1)
#     return torch.load(io.BytesIO(bytes(payload.tolist())))  # 还原为 bytes 再反序列化

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
TAG_FSCAT = 70_000
TAG_BSCAT = 80_000
TAG_FWD_A2A = 90_000
TAG_BWD_A2A = 100_000
def tag_fwd(mb): return TAG_BASE_FWD + mb
def tag_bwd(mb): return TAG_BASE_BWD + mb
def tag_est(mb, q): return TAG_EST + (mb * 1000) + q
def tag_lbl_last(mb): return TAG_LBL_LAST + mb
def tag_lbl_z(mb):    return TAG_LBL_Z + mb
def tag_loss():       return TAG_LOSS
def tag_fwd_scatter(mb): return TAG_FSCAT + mb
def tag_bwd_scatter(mb): return TAG_BSCAT + mb
def tag_fwd_a2a(mb, src_rid): return TAG_FWD_A2A + (mb * 1024) + src_rid
def tag_bwd_a2a(mb, src_rid): return TAG_BWD_A2A + (mb * 1024) + src_rid

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
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/nvidia/Llama-3.2-3B_old/",
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
        model_config = AutoConfig.from_pretrained("/home/nvidia/Llama-3.2-3B_old/")
        tmp_model = AutoModelForCausalLM.from_pretrained(
            "/home/nvidia/Llama-3.2-3B_old/",
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

def _build_hpp_topology(model_split_configs, stage_count_ls, stage_count_id_ls, iszeroth_ls):
    keys = []
    for r, cfg in enumerate(model_split_configs):
        start, end = cfg[0], cfg[1]
        keys.append((r, start, end))
    keys.sort(key=lambda x: (x[1], x[2]))
    uniq = []
    last = None
    for _, st, ed in keys:
        if last is None or (st, ed) != last:
            uniq.append((st, ed))
            last = (st, ed)

    groups = []
    group_of_rank = {}
    rid_of_rank = {}
    g_iszeroth = []
    for g_idx, (st, ed) in enumerate(uniq):
        ranks_in_g = [r for r, (s, e) in ((i, (model_split_configs[i][0], model_split_configs[i][1]))
                                          for i in range(len(model_split_configs))) if (s, e) == (st, ed)]
        ranks_in_g.sort(key=lambda r: stage_count_id_ls[r])
        for r in ranks_in_g:
            group_of_rank[r] = g_idx
            rid_of_rank[r] = stage_count_id_ls[r]
        groups.append(ranks_in_g)
        g_iszeroth.append(bool(iszeroth_ls[ranks_in_g[0]]))

    K = [len(ranks) for ranks in groups]
    def rank_of(g, rid): return groups[g][rid]
    return groups, group_of_rank, rid_of_rank, K, g_iszeroth, rank_of

def _make_stage_process_groups(groups):
    return [dist.new_group(ranks=ranks) for ranks in groups]

def allreduce_trainable_grads(module, group, world_size_in_group: int):
    handles = []
    for p in module.parameters():
        if p.requires_grad and p.grad is not None:
            handles.append(dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, group=group, async_op=True))
    for h in handles: h.wait()
    for p in module.parameters():
        if p.requires_grad and p.grad is not None:
            p.grad.div_(world_size_in_group)

def _balanced_sizes(total, K):
    # 允许不能整除，余数分配到前面几个分片
    base = total // K
    rem = total % K
    sizes = [(base + 1 if i < rem else base) for i in range(K)]
    offs = [0]
    for sz in sizes:
        offs.append(offs[-1] + sz)
    return sizes, offs[:-1]  # sizes, offsets(起点)

def _slice_obj_batch(obj, start, end):
    if torch.is_tensor(obj):
        return obj[start:end]

    elif isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "position_embeddings":
                # v 预期为 (cos, sin) 或 None
                if v is None:
                    out[k] = None
                else:
                    cos, sin = v
                    # 若 cos/sin 的 batch 维为 1（广播形状），不按 batch 切，原样传递；
                    # 否则按 batch 切。
                    if torch.is_tensor(cos) and cos.dim() >= 1 and cos.shape[0] > 1:
                        out[k] = (cos[start:end], sin[start:end])
                    else:
                        out[k] = (cos, sin)
            else:
                out[k] = _slice_obj_batch(v, start, end)
        return out

    elif isinstance(obj, (list, tuple)):
        return type(obj)(_slice_obj_batch(x, start, end) for x in obj)

    else:
        return obj

def _concat_objs(objs):
    objs = [o for o in objs if o is not None]
    if len(objs) == 0:
        return None

    p0 = objs[0]

    if torch.is_tensor(p0):
        return torch.cat(objs, dim=0) if len(objs) > 1 else p0

    elif isinstance(p0, dict):
        out = {}
        # 收集所有 key，兼容不同分片缺少某 key 的情况
        keys = set().union(*[o.keys() for o in objs])
        for k in keys:
            if k == "position_embeddings":
                parts = []
                for o in objs:
                    if k not in o or o[k] is None:
                        continue
                    cos, sin = o[k]
                    # 该分片对应的 hidden_states batch 尺寸（用于对齐）
                    hs = o.get("hidden_states", None)
                    b_part = int(hs.shape[0]) if (isinstance(hs, torch.Tensor) and hs.ndim > 0) else None
                    if b_part is None or b_part == 0:
                        continue
                    # 如果 cos/sin 的 batch 维为 1（广播），拉伸到 b_part
                    if cos.shape[0] == 1 and b_part > 1:
                        cos = cos.expand(b_part, *cos.shape[1:])
                        sin = sin.expand(b_part, *sin.shape[1:])
                    elif cos.shape[0] != b_part:
                        raise RuntimeError(
                            f"position_embeddings part batch {cos.shape[0]} != hidden_states part batch {b_part}"
                        )
                    parts.append((cos, sin))
                if len(parts) == 0:
                    out[k] = None
                else:
                    cos_cat = torch.cat([c for c, _ in parts], dim=0)
                    sin_cat = torch.cat([s for _, s in parts], dim=0)
                    out[k] = (cos_cat, sin_cat)
            else:
                # 普通 key：递归拼接（注意过滤缺失该 key 的分片）
                out[k] = _concat_objs([o[k] for o in objs if k in o])
        return out

    elif isinstance(p0, (list, tuple)):
        trans = list(zip(*objs))
        return type(p0)(_concat_objs(list(x)) for x in trans)

    else:
        return p0

def _global_ranges(total, K):
    sizes, offs = _balanced_sizes(total, K)
    ranges = [(offs[i], offs[i]+sizes[i]) for i in range(K)]
    return ranges, sizes

def _reshard_send_to_next(part_local, B_total, K_src, src_rid, K_dst, g, mb, rank_of):
    """
    本组副本 src_rid 把自己这份 (global区间=src_ranges[src_rid]) 按 K_dst 切成多个子段，
    直接发给下一组的每个目标副本。
    """
    src_ranges, _ = _global_ranges(B_total, K_src)
    dst_ranges, _ = _global_ranges(B_total, K_dst)
    src_s, src_e = src_ranges[src_rid]
    # 遍历目标副本，算交集再发送
    for dst_rid, (dst_s, dst_e) in enumerate(dst_ranges):
        s = max(src_s, dst_s)
        e = min(src_e, dst_e)
        if s >= e:   # 无交集
            # 也可以不发，接收端按固定顺序收 K_src 份（允许是空对象）
            send_obj({"empty": True}, dst=rank_of(g+1, dst_rid), tag=tag_fwd_a2a(mb, src_rid))
            continue
        # 映射到本地局部切片
        local_s, local_e = s - src_s, e - src_s
        seg = _slice_obj_batch(part_local, local_s, local_e)
        send_obj({"empty": False, "seg": seg}, dst=rank_of(g+1, dst_rid), tag=tag_fwd_a2a(mb, src_rid))

def _reshard_recv_from_prev(B_total, K_src, K_dst, dst_rid, g, mb, rank_of):
    """
    本组副本 dst_rid 从上一组的每个 src_rid 接收一段（可能为空），
    按 src_rid 升序拼接，得到自己的局部分片。
    """
    segs = []
    for src_rid in range(K_src):
        pkt = recv_obj(src=rank_of(g-1, src_rid), tag=tag_fwd_a2a(mb, src_rid))
        # print(pkt)
        if not isinstance(pkt, dict) or pkt.get("empty", False):
            continue
        segs.append(pkt["seg"])
    # print(segs)
    return _concat_objs(segs)

def _reshard_send_grad_to_prev(grad_part, B_total, K_src, K_dst, dst_rid, g, mb, rank_of):
    """
    反向：本组（g+1）的 dst_rid 把自己这份梯度，按上一组的 K_src 切成多段，直发上一组对应 src_rid。
    """
    dst_ranges, _ = _global_ranges(B_total, K_dst)
    src_ranges, _ = _global_ranges(B_total, K_src)
    dst_s, dst_e = dst_ranges[dst_rid]
    for src_rid, (src_s, src_e) in enumerate(src_ranges):
        s = max(src_s, dst_s)
        e = min(src_e, dst_e)
        if s >= e:
            send_obj({"empty": True}, dst=rank_of(g-1, src_rid), tag=tag_bwd_a2a(mb, dst_rid))
            continue
        # 映射到本地局部切片（注意这里基于 dst 的本地坐标）
        local_s, local_e = s - dst_s, e - dst_s
        seg = _slice_obj_batch(grad_part, local_s, local_e)
        send_obj({"empty": False, "seg": seg}, dst=rank_of(g-1, src_rid), tag=tag_bwd_a2a(mb, dst_rid))

def _reshard_recv_grad_from_next(B_total, K_src, K_dst, src_rid, g, mb, rank_of):
    """上一组的 src_rid 从下一组每个 dst_rid 收梯度分段，按 dst_rid 升序拼起来"""
    segs = []
    for dst_rid in range(K_dst):
        pkt = recv_obj(src=rank_of(g+1, dst_rid), tag=tag_bwd_a2a(mb, dst_rid))
        if not isinstance(pkt, dict) or pkt.get("empty", False):
            continue
        segs.append(pkt["seg"])
    return _concat_objs(segs)

# ===== PS 风格的按参数名聚合 =====
TAG_PS_META      = 150_000   # 注册本地可训练参数名
TAG_PS_PUSH_BASE = 151_000   # 向 PS 发送梯度（按 iter_idx 叠加）
TAG_PS_PULL_BASE = 152_000   # PS 回发平均梯度（按 iter_idx 叠加）
def tag_ps_push(iter_idx): return TAG_PS_PUSH_BASE + iter_idx
def tag_ps_pull(iter_idx): return TAG_PS_PULL_BASE + iter_idx

def _trainable_param_names_and_shapes(module):
    # 仅收集 requires_grad=True 的参数（你现在只训练 LoRA）
    names, shapes, dtypes = [], [], []
    for n, p in module.named_parameters():
        if p.requires_grad:
            names.append(n)
            shapes.append(tuple(p.shape))
            dtypes.append(str(p.dtype))
    return names, shapes, dtypes

def _ps_register_owner_map(sub_network, ps_rank, world_size):
    """
    每个 rank 把自己本地可训练参数名列表发给 PS；
    PS 汇总出 owner_map: {param_name: [ranks that own it]} 和 per_rank_names: {rank: [names]}。
    最后 PS 广播 owner_map 给所有人，返回 (owner_map, my_names)。
    """
    rank = dist.get_rank()
    my_names, _, _ = _trainable_param_names_and_shapes(sub_network)
    if rank == ps_rank:
        per_rank_names = {ps_rank: my_names}
        # 收集其他 rank 的参数名
        for r in range(world_size):
            if r == ps_rank: 
                continue
            names_r = recv_obj(src=r, tag=TAG_PS_META)
            per_rank_names[r] = names_r
        # 构造 owner_map
        owner_map = {}
        for r, lst in per_rank_names.items():
            for n in lst:
                owner_map.setdefault(n, []).append(r)
        # 广播 owner_map
        obj = [owner_map]
        dist.broadcast_object_list(obj, src=ps_rank)
        return obj[0], my_names
    else:
        send_obj(my_names, dst=ps_rank, tag=TAG_PS_META)
        obj = [None]
        dist.broadcast_object_list(obj, src=ps_rank)
        return obj[0], my_names

def _ps_allreduce_grads_once(sub_network, owner_map, my_names, ps_rank, iter_idx):
    """
    一轮迭代的反向结束后调用。
    - 非 PS：推送本地 grads[name] 给 PS -> 阻塞等待回包 -> 覆盖 .grad
    - PS   ：收集所有 grads -> 对同名参数平均 -> 直接写回自己本地 -> 给其它拥有者回发平均梯度
    """
    rank = dist.get_rank()
    world = dist.get_world_size()

    if rank != ps_rank:
        # 1) 组装并发送我的梯度（允许为空）
        grads = {}
        for n, p in sub_network.named_parameters():
            if p.requires_grad and (n in my_names) and (p.grad is not None):
                grads[n] = p.grad.detach().cpu()
        send_obj(grads, dst=ps_rank, tag=tag_ps_push(iter_idx))
        flush_sends()  # 立刻冲掉，避免句柄积压

        # 2) 等待 PS 回包（可能是空 dict）
        avg_grads = recv_obj(src=ps_rank, tag=tag_ps_pull(iter_idx))

        # 3) 写回（只写我拥有且在回包里的项）
        name_to_param = dict(sub_network.named_parameters())
        for n, g in avg_grads.items():
            if n in name_to_param:
                param = name_to_param[n]
                if param.grad is None:
                    param.grad = torch.zeros_like(param, dtype=param.dtype, device=param.device)
                param.grad.copy_(g.to(param.grad.device, dtype=param.grad.dtype))
        return

    # ====== PS 逻辑 ======
    # 1) 统计拥有者数量
    counts = {name: len(owners) for name, owners in owner_map.items()}

    # 2) 自己的梯度
    sums = {}
    for n, p in sub_network.named_parameters():
        if p.requires_grad and (p.grad is not None):
            g = p.grad.detach().cpu()
            sums[n] = g.clone() if n not in sums else (sums[n] + g)

    # 3) 收集其他 rank 的梯度并累加（允许对方发空字典）
    for r in range(world):
        if r == ps_rank:
            continue
        grads_r = recv_obj(src=r, tag=tag_ps_push(iter_idx))
        for n, g in grads_r.items():
            if n in sums:
                sums[n] += g
            else:
                sums[n] = g.clone()

    # 4) 计算平均，并直接回写到本地（PS 自己），同时准备给其它 rank 的回包
    #    只给拥有该参数的 rank 回发对应条目
    reply = {r: {} for r in range(world)}
    name_to_param = dict(sub_network.named_parameters())

    for n, tensor_sum in sums.items():
        c = counts.get(n, 1)
        avg = tensor_sum / float(c)

        # 回写到 PS 自己（如果拥有）
        if n in name_to_param:
            param = name_to_param[n]
            if param.grad is None:
                param.grad = torch.zeros_like(param, dtype=param.dtype, device=param.device)
            param.grad.copy_(avg.to(param.grad.device, dtype=param.grad.dtype))

        # 给其它拥有者准备回包（排除自己）
        for r in owner_map.get(n, []):
            if r != ps_rank:
                reply[r][n] = avg  # 仍然先放 CPU，接收端再 to(device)

    # 5) 把回包发给其它 rank（自己不发）并 flush
    for r in range(world):
        if r == ps_rank:
            continue
        send_obj(reply[r], dst=r, tag=tag_ps_pull(iter_idx))
    flush_sends()



def run_training(args):
    rank = dist.get_rank()
    world = dist.get_world_size()

    # 1) 分段（所有 rank 都跑一遍，得到一致结果）
    model_split_configs, iszeroth_ls, device_to_stage_ls, stage_to_device_ls, stage_count_ls, stage_count_id_ls = stage_partition(
        args.training_methods, args.model_name_or_path
    )

    if args.training_methods == "HPP":
        groups, group_of_rank, rid_of_rank, K, g_iszeroth, rank_of = _build_hpp_topology(
            model_split_configs, stage_count_ls, stage_count_id_ls, iszeroth_ls
        )
        G = len(groups)
        g = group_of_rank[rank]
        rid = rid_of_rank[rank]
        last_group = G - 1
        stage_pgs = _make_stage_process_groups(groups)
        my_pg = stage_pgs[g]

        # 数据与标签：组0的 leader 读取、切 M；标签按 K[last_group] 下发
        leader0 = rank_of(0, 0)
        if g == 0 and rank == leader0:
            train_dataloader, args = load_train_dataloader(args)
            meta = {
                "total_iterations": len(train_dataloader),
                "micro_batch_num": int(args.batch_size // args.micro_batch_size),
                "vocab_size": args.vocab_size,
                "max_seq_len": args.max_seq_len,
            }
        else:
            train_dataloader, meta = None, None
        obj = [meta]
        dist.broadcast_object_list(obj, src=leader0)
        meta = obj[0]
        if not (g == 0 and rank == leader0):
            args.total_iterations = meta["total_iterations"]
            args.micro_batch_num  = meta["micro_batch_num"]
            args.vocab_size       = meta["vocab_size"]
            args.max_seq_len      = meta["max_seq_len"]

        M = args.micro_batch_num
        B = args.micro_batch_size
        sub = DistributedSubModel(args, model_split_configs[rank], iszeroth_ls,
                                stage_idx=g, iszeroth=iszeroth_ls[rank], total_stages=G)
        data_iter = iter(train_dataloader) if (g == 0 and rank == leader0) else None

        base_forward_total_steps  = G + M - 1
        base_backward_total_steps = G + M - 1
        total_steps = base_forward_total_steps + base_backward_total_steps
        for epoch_idx in range(args.epochs):
            if rank == groups[0][0]:
                print(f"[epoch {epoch_idx}] iters={args.total_iterations}", flush=True)
    
            for iter_idx in range(args.total_iterations):
                sub.set_train_configuration()

                # 组0 leader 拉数据并切 micro-batch；标签按 K[last_group] 切发给最后一组各副本
                if g == 0 and rank == leader0:
                    batch_inputs, batch_labels = next(data_iter)
                    micro_input_ids_ls = torch.split(batch_inputs["input_ids"], B, dim=0)
                    micro_attention_mask_ls = torch.split(batch_inputs["attention_mask"], B, dim=0)
                    micro_labels = torch.split(batch_labels, B, dim=0)
                    micro_inputs = [
                        dict(input_ids=ids, attention_mask=attn)
                        for ids, attn in zip(micro_input_ids_ls, micro_attention_mask_ls)
                    ]
                    # 标签直接按 K[last_group] 细分后发到各副本（每个 mb 发 K[last_group] 份）
                    sizes_last, offs_last = _balanced_sizes(B, K[last_group])
                    for mb in range(M):
                        for rlast in range(K[last_group]):
                            s = offs_last[rlast]; e = s + sizes_last[rlast]
                            send_obj(_slice_obj_batch(micro_labels[mb], s, e),
                                    dst=rank_of(last_group, rlast), tag=tag_lbl_last(mb))
                else:
                    if g == last_group:
                        sub.labels = {mb: recv_obj(src=leader0, tag=tag_lbl_last(mb)) for mb in range(M)}
                flush_sends()
                dist.barrier()

                if rank == leader0:
                    start = time.time()

                for step_idx in range(total_steps):
                    # ====== Forward ======
                    if step_idx < base_forward_total_steps:
                        mb = step_idx - g
                        if 0 <= mb < M:
                            # a) 获取“进入本组”的本地 part
                            if g == 0:
                                # 组0：leader把完整 micro-batch 按 K[0] 切，直发到各副本
                                sizes0, offs0 = _balanced_sizes(B, K[0])
                                if rank == leader0:
                                    for r0 in range(K[0]):
                                        s = offs0[r0]; e = s + sizes0[r0]
                                        part = _slice_obj_batch(micro_inputs[mb], s, e)
                                        if r0 == 0:
                                            local_in = part
                                        else:
                                            send_obj(part, dst=rank_of(0, r0), tag=tag_fwd_scatter(mb))
                                else:
                                    local_in = recv_obj(src=leader0, tag=tag_fwd_scatter(mb))
                            else:
                                # 其它组：直接 all-to-all 从上一组各副本接收（重分片）
                                local_in = _reshard_recv_from_prev(B, K[g-1], K[g], rid, g, mb, rank_of)

                            # b) 本副本前向
                            out_part = sub.base_forward_only(epoch_idx, iter_idx, mb, inputs=local_in)

                            # c) 若非最后一组：本副本把 out_part 直接“重分片”发到下一组各副本
                            if g < last_group:
                                _reshard_send_to_next(out_part, B, K[g], rid, K[g+1], g, mb, rank_of)

                    # ====== Backward ======
                    else:
                        inv = last_group - g
                        mb = step_idx - base_forward_total_steps - inv
                        if 0 <= mb < M:
                            if g == last_group:
                                # 最后一组：各副本直接用自己的 label 切片计算 & 产生对上一组的梯度分片
                                grad_part = sub.base_backward_only(mb)
                                if K[g] > 1:
                                    allreduce_trainable_grads(sub.network, stage_pgs[g], K[g])
                                _reshard_send_grad_to_prev(grad_part, B, K[g-1], K[g], rid, g, mb, rank_of)
                            else:
                                # 其它组：先从下一组各副本收“对本副本的梯度分片”，拼成本地梯度，再反传
                                grad_in_part = _reshard_recv_grad_from_next(B, K[g], K[g+1], rid, g, mb, rank_of)
                                grad_part = sub.base_backward_only(mb, grad_in_part)
                                if K[g] > 1:
                                    allreduce_trainable_grads(sub.network, stage_pgs[g], K[g])
                                if g > 0:
                                    _reshard_send_grad_to_prev(grad_part, B, K[g-1], K[g], rid, g, mb, rank_of)

                flush_sends()
                dist.barrier()
                sub.network_optimizer()

                if rank == leader0:
                    end = time.time()
                    print(f"[iter {iter_idx}] done time: {end-start}.")
                    
    elif args.training_methods == "HDP":
        # ---------- 1) 构造 DP 组的“完整流水线” ----------
        # 先按 (layer_start, layer_end) 给每个 rank 标注 stage 顺序
        key_per_rank = [(i, model_split_configs[i][0], model_split_configs[i][1]) for i in range(len(model_split_configs))]
        key_per_rank.sort(key=lambda x: (x[1], x[2]))
        uniq_stages = []
        last = None
        for _, st, ed in key_per_rank:
            if last is None or (st, ed) != last:
                uniq_stages.append((st, ed))
                last = (st, ed)
        S = len(uniq_stages)

        # 每个 stage 里按 stage_count_id 升序得到该 stage 的 DP 副本列表
        ranks_per_stage = []
        for s_idx, (st, ed) in enumerate(uniq_stages):
            rs = [r for r,(s,e) in ((i,(model_split_configs[i][0], model_split_configs[i][1])) for i in range(len(model_split_configs))) if (s,e)==(st,ed)]
            rs.sort(key=lambda r: stage_count_id_ls[r])   # 用你已有的“副本序号”
            ranks_per_stage.append(rs)
        # DP 组数（要求每个 stage 的副本数一致）
        K_dp = len(ranks_per_stage[0])
        assert all(len(ranks_per_stage[s]) == K_dp for s in range(S)), \
            f"HDP 需要每个 stage 在所有 DP 组中都有一个副本；各 stage 副本数={[len(ranks_per_stage[s]) for s in range(S)]}"

        # 第 gid 组的流水线 = [ranks_per_stage[0][gid], ranks_per_stage[1][gid], ...]
        pipelines = [[ranks_per_stage[s][gid] for s in range(S)] for gid in range(K_dp)]
        # 当前 rank 所处的 DP 组及其在组内的 stage 索引
        def _find_gid_and_s():
            for gid in range(K_dp):
                for s in range(S):
                    if pipelines[gid][s] == rank:
                        return gid, s
            raise RuntimeError("rank not found in pipelines")
        gid, s_local = _find_gid_and_s()
        pipe = pipelines[gid]
        last_stage = S - 1

        # ---------- 2) 选择 PS / Loader ----------
        ps_rank = pipelines[0][0]      # 组0的 stage0 既做 loader 又做 PS
        global_loader = ps_rank

        # ---------- 3) 本 rank 子模型 ----------
        sub = DistributedSubModel(args, model_split_configs[rank], [False]*len(model_split_configs),
                                stage_idx=s_local, iszeroth=False, total_stages=S)

        # ---------- 4) 注册参数所有者映射（owner_map） ----------
        owner_map, my_names = _ps_register_owner_map(sub.network, ps_rank, dist.get_world_size())

        # ---------- 5) 数据/标签：集中分发（按 DP 维切 K 份） ----------
        if rank == global_loader:
            train_dataloader, args = load_train_dataloader(args)
            meta = {
                "total_iterations": len(train_dataloader),
                "micro_batch_num": int(args.batch_size // args.micro_batch_size),
                "vocab_size": args.vocab_size,
                "max_seq_len": args.max_seq_len,
            }
        else:
            train_dataloader, meta = None, None
        obj = [meta]
        dist.broadcast_object_list(obj, src=global_loader)
        meta = obj[0]
        if rank != global_loader:
            args.total_iterations = meta["total_iterations"]
            args.micro_batch_num  = meta["micro_batch_num"]
            args.vocab_size       = meta["vocab_size"]
            args.max_seq_len      = meta["max_seq_len"]

        M = args.micro_batch_num
        B = args.micro_batch_size
        sizes_dp, offs_dp = _balanced_sizes(B, K_dp)

        prev_rank = pipe[s_local - 1] if s_local > 0 else None
        next_rank = pipe[s_local + 1] if s_local < last_stage else None

        data_iter = iter(train_dataloader) if (rank == global_loader) else None

        base_forward_total_steps  = S + M - 1
        base_backward_total_steps = S + M - 1
        total_steps = base_forward_total_steps + base_backward_total_steps

        for epoch_idx in range(args.epochs):
            if rank == global_loader:
                print(f"[epoch {epoch_idx}] iters={args.total_iterations}", flush=True)

            for iter_idx in range(args.total_iterations):
                sub.set_train_configuration()

                # ===== 分发本轮 micro-batch 的 inputs/labels 到各 DP 组 =====
                if rank == global_loader:
                    batch_inputs, batch_labels = next(data_iter)
                    micro_input_ids_ls = torch.split(batch_inputs["input_ids"], B, dim=0)
                    micro_attention_mask_ls = torch.split(batch_inputs["attention_mask"], B, dim=0)
                    micro_labels = torch.split(batch_labels, B, dim=0)
                    mb_inputs = [dict(input_ids=ids, attention_mask=attn)
                                for ids, attn in zip(micro_input_ids_ls, micro_attention_mask_ls)]
                    # 逐 DP 组切片并发送
                    local_inputs_gid0 = {}
                    for mb in range(M):
                        for gidx in range(K_dp):
                            s0 = offs_dp[gidx]; e0 = s0 + sizes_dp[gidx]
                            inp_slice = _slice_obj_batch(mb_inputs[mb], s0, e0)
                            lbl_slice = _slice_obj_batch(micro_labels[mb], s0, e0)
                            stage0_r = pipelines[gidx][0]
                            last_r   = pipelines[gidx][last_stage]
                            if gidx == 0:
                                local_inputs_gid0[mb] = inp_slice
                            else:
                                send_obj(inp_slice, dst=stage0_r, tag=tag_fwd_scatter(mb))
                            send_obj(lbl_slice, dst=last_r,   tag=tag_lbl_last(mb))
                    if gid == 0 and s_local == 0:
                        micro_inputs_local = [local_inputs_gid0[m] for m in range(M)]
                else:
                    if s_local == 0:
                        micro_inputs_local = [recv_obj(src=global_loader, tag=tag_fwd_scatter(m)) for m in range(M)]
                    if s_local == last_stage:
                        sub.labels = {m: recv_obj(src=global_loader, tag=tag_lbl_last(m)) for m in range(M)}

                flush_sends(); dist.barrier()

                # ===== 组内纯流水线调度 =====
                for step_idx in range(total_steps):
                    # ---------- Forward ----------
                    if step_idx < base_forward_total_steps:
                        mb = step_idx - s_local
                        if 0 <= mb < M:
                            if s_local == 0:
                                out = sub.base_forward_only(epoch_idx, iter_idx, mb, inputs=micro_inputs_local[mb])
                            else:
                                inp = recv_obj(src=prev_rank, tag=tag_fwd(mb))
                                out = sub.base_forward_only(epoch_idx, iter_idx, mb, inputs=inp)
                            if s_local < last_stage:
                                send_obj(out, dst=next_rank, tag=tag_fwd(mb))
                    # ---------- Backward ----------
                    else:
                        inv = last_stage - s_local
                        mb = step_idx - base_forward_total_steps - inv
                        if 0 <= mb < M:
                            if inv == 0:
                                grad = sub.base_backward_only(mb)
                                if s_local > 0:
                                    send_obj(grad, dst=prev_rank, tag=tag_bwd(mb))
                            else:
                                grad_in = recv_obj(src=next_rank, tag=tag_bwd(mb))
                                grad = sub.base_backward_only(mb, grad_in)
                                if s_local > 0:
                                    send_obj(grad, dst=prev_rank, tag=tag_bwd(mb))

                flush_sends(); dist.barrier()

                # =====★ 本轮结束：PS 聚合梯度（按参数名） =====
                _ps_allreduce_grads_once(sub.network, owner_map, my_names, ps_rank, iter_idx)

                dist.barrier()
                sub.network_optimizer()
    elif args.training_methods == "mobius":
        # TODO: 你的现有 Mobius 流水线
        pass


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
        timeout=timedelta(minutes=2),
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
