#!/usr/bin/env bash
set -euo pipefail

# ====== 必填：集群主机列表（第一个作为 Master）======
HOSTS=("nodeA" "nodeB")   # 改成你的主机名或IP
PORT=29001                # 你传的 --master-port
WORKDIR="$(pwd)"          # 或指定你的代码目录

# ====== 通用设置（单卡/机）======
WORLD_SIZE=${#HOSTS[@]}
MASTER_ADDR="${HOSTS[0]}"
# ENV_EXPORTS="\
# NCCL_DEBUG=INFO \
# NCCL_ASYNC_ERROR_HANDLING=1 \
# OMP_NUM_THREADS=1 \
# CUDA_VISIBLE_DEVICES=0 \
# NCCL_SOCKET_IFNAME=eth0"   # 按实际网卡改，如 ib0/ens3 等

# ====== 你的训练命令（原样放进来）======
TRAIN_CMD="python mm_hybrid_v1_dist.py \
  --q_num 2 \
  --zo_step_size 1e-3 \
  --batch_size 16 \
  --micro_batch_size 4 \
  --max_seq_len 512 \
  --training_methods hybrid \
  --lr 2e-5 \
  --master-port ${PORT}"

# ====== 启动所有节点（并发）======
for i in "${!HOSTS[@]}"; do
  host="${HOSTS[$i]}"
  echo "[LAUNCH] $host  (RANK=$i / WORLD_SIZE=${WORLD_SIZE})"
  ssh -o StrictHostKeyChecking=no "$host" "
    set -e
    cd ${WORKDIR}
    export MASTER_ADDR=${MASTER_ADDR}
    export MASTER_PORT=${PORT}
    export RANK=${i}
    export WORLD_SIZE=${WORLD_SIZE}
    export LOCAL_RANK=0
    ${TRAIN_CMD}
  " &
done

wait
echo "All nodes started."
