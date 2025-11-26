import os, sys, time, argparse
from datetime import timedelta
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import TCPStore

sys.stdout.reconfigure(line_buffering=True)
def log(*a): print(*a, flush=True)

# ---------- 模型分段（示例：2 段） ----------
class Part1(nn.Module):
    def __init__(self, in_f=256, hid=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_f, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
        )
    def forward(self, x): return self.net(x)

class Part2(nn.Module):
    def __init__(self, hid=512, out_f=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, out_f)
        )
    def forward(self, x): return self.net(x)

# ---------- 调度：master（eth0 上的 TCPStore） ----------
def master_ctrl(ctrl_addr, ctrl_port, data_addr, data_port, micro_batches=8):
    store = TCPStore(ctrl_addr, ctrl_port, world_size=3, is_master=True,
                     wait_for_workers=True, timeout=timedelta(seconds=60))
    log("[MASTER] store up on", ctrl_addr, ctrl_port)

    # 等两个 worker 报到
    store.wait(["w1_ready", "w2_ready"], timedelta(seconds=60))
    log("[MASTER] workers joined")

    # 下发数据面 rendezvous 与批次数
    store.set("data_addr", data_addr.encode())
    store.set("data_port", str(data_port).encode())
    store.set("M", str(micro_batches).encode())

    # 让两端初始化数据面 PG
    store.set("evt_INIT_PG", b"1")
    store.wait(["w1_pg_up", "w2_pg_up"], timedelta(seconds=60))
    log("[MASTER] data PG up")

    # 自检（all_reduce==2.0）
    store.set("evt_SANITY", b"1")
    store.wait(["w1_sane", "w2_sane"], timedelta(seconds=60))
    log("[MASTER] sanity OK")

    # 开始流水线
    store.set("evt_START", b"1")
    log("[MASTER] pipeline started")

    # 等两端结束
    store.wait(["w1_done_all", "w2_done_all"], timedelta(seconds=600))
    log("[MASTER] pipeline finished")

    # 回收
    store.set("evt_SHUTDOWN", b"1")
    log("[MASTER] shutdown sent")

# ---------- 工作者：eth1 上的数据面 + 1F1B 流水线 ----------
def worker(ctrl_addr, ctrl_port, role, data_iface, in_f=256, hid=512, out_f=10):
    assert role in ("worker1", "worker2")
    store = TCPStore(ctrl_addr, ctrl_port, world_size=3, is_master=False,
                     wait_for_workers=True, timeout=timedelta(seconds=60))
    store.set("w1_ready" if role=="worker1" else "w2_ready", b"1")
    log(f"[{role}] connected to store {ctrl_addr}:{ctrl_port}")

    # 等 master 下发
    store.wait(["evt_INIT_PG"], timedelta(seconds=60))
    data_addr = store.get("data_addr").decode()
    data_port = int(store.get("data_port").decode())
    M = int(store.get("M").decode())

    # ----- 初始化数据面 PG（仅两 worker；走 eth1） -----
    os.environ["GLOO_SOCKET_IFNAME"] = data_iface
    data_rank = 0 if role=="worker1" else 1
    log(f"[{role}][DATA] init PG via {data_iface} -> {data_addr}:{data_port} rank={data_rank}/2")
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://{data_addr}:{data_port}",
        rank=data_rank, world_size=2, timeout=timedelta(seconds=30)
    )
    pg = dist.new_group(ranks=[0,1])
    store.set("w1_pg_up" if role=="worker1" else "w2_pg_up", b"1")

    # ----- 模型与优化器（CPU 演示；要 GPU 请见文末） -----
    if role == "worker1":
        part = Part1(in_f, hid)
        opt  = torch.optim.SGD(part.parameters(), lr=1e-2)
    else:
        part = Part2(hid, out_f)
        opt  = torch.optim.SGD(part.parameters(), lr=1e-2)

    # 自检：all_reduce==2.0
    store.wait(["evt_SANITY"], timedelta(seconds=60))
    x = torch.tensor([1.0])
    dist.all_reduce(x, op=dist.ReduceOp.SUM, group=pg)
    (store.set("w1_sane", b"1") if role=="worker1" else store.set("w2_sane", b"1"))
    log(f"[{role}] sanity={x.item()} (expect 2.0)")

    # 数据准备（演示用随机数据/标签）
    # worker1 生成输入，worker2 生成标签
    batch = [torch.randn(32, in_f) for _ in range(M)] if role=="worker1" else None
    labels = [torch.randint(0, out_f, (32,)) for _ in range(M)] if role=="worker2" else None
    ce = nn.CrossEntropyLoss()

    # 事件：开始流水线
    store.wait(["evt_START"], timedelta(seconds=300))

    # ----- 1F1B 调度（2 段） -----
    # tag 设计：FWD = 1000 + mb；BWD = 2000 + mb
    def fwd_tag(mb): return 1000 + mb
    def bwd_tag(mb): return 2000 + mb

    if role == "worker1":
        # ----- 1F1B：worker1（stage 0）-----
        acts_cache = {}
        for mb in range(M):
            # 1) 除了第 0 个，其余都先接收上一个 mb 的梯度并反向
            if mb > 0:
                grad = torch.empty_like(acts_cache[mb-1])
                dist.recv(grad, src=1, tag=bwd_tag(mb-1))     # 先收
                log(f"[worker1] BWD recv grad mb{mb-1}")
                opt.zero_grad(set_to_none=True)
                acts_cache[mb-1].backward(grad)               # 直接对边界激活回传
                opt.step()
                del acts_cache[mb-1]

            # 2) 当前 mb 前向并发送激活
            x_mb = batch[mb]
            act = part(x_mb)
            acts_cache[mb] = act                               # 不要 detach
            dist.send(act, dst=1, tag=fwd_tag(mb))             # 再发
            log(f"[worker1] FWD sent mb{mb}")

        # 3) drain：处理最后一个 mb 的反向
        grad_last = torch.empty_like(acts_cache[M-1])
        dist.recv(grad_last, src=1, tag=bwd_tag(M-1))
        log(f"[worker1] BWD recv grad mb{M-1}")
        opt.zero_grad(set_to_none=True)
        acts_cache[M-1].backward(grad_last)
        opt.step()
        del acts_cache[M-1]
        store.set("w1_done_all", b"1")
    else:  # worker2
        # 缓存边界激活，用于 loss 和反向
        acts = {}
        for mb in range(M):
            act = torch.empty(32, hid)
            dist.recv(act, src=0, tag=fwd_tag(mb))
            act.requires_grad_(True)               # 必须！
            logits = part(act)
            loss = ce(logits, labels[mb])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            grad_to_prev = act.grad.detach().clone()
            dist.send(grad_to_prev, dst=0, tag=bwd_tag(mb))
            log(f"[{role}] BWD sent grad mb{mb}")

        store.set("w2_done_all", b"1")
    log(f"[{role}] done all")

    # 回收
    dist.destroy_process_group(group=pg)
    dist.destroy_process_group()

    # 等 master 发关机事件（可选）
    # store.wait(["evt_SHUTDOWN"], timedelta(seconds=300))
    log(f"[{role}] exit")

# ---------- 入口 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["master","worker1","worker2"], required=True)

    # 控制面：TCPStore（走 eth0）
    ap.add_argument("--ctrl-master-addr", default="192.168.1.125")
    ap.add_argument("--ctrl-port", type=int, default=29400)

    # 数据面：PG rendezvous（走 worker1 的 eth1）
    ap.add_argument("--data-master-addr", default="192.168.137.222")
    ap.add_argument("--data-port", type=int, default=29501)
    ap.add_argument("--data-iface", default="eth1")

    ap.add_argument("--micro-batches", type=int, default=8)
    args = ap.parse_args()

    if args.role == "master":
        master_ctrl(args.ctrl_master_addr, args.ctrl_port,
                    args.data_master_addr, args.data_port,
                    args.micro_batches)
    elif args.role == "worker1":
        worker(args.ctrl_master_addr, args.ctrl_port, "worker1", args.data_iface)
    else:
        worker(args.ctrl_master_addr, args.ctrl_port, "worker2", args.data_iface)

if __name__ == "__main__":
    main()
