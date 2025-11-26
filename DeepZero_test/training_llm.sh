#!/bin/bash

# 设置变量
q_num=3
learning_rate=("1e-5")
gpus='0,1,2,5'
zo_step_size=1e-3
batch_size=16
micro_batch_size=4
max_seq_len=1500
training_methods="only_first"

other_args="conda activate zero && python -u hybrid_trianing_llm.py --q_num $q_num --zo_step_size $zo_step_size --batch_size $batch_size --micro_batch_size $micro_batch_size --max_seq_len $max_seq_len --training_methods $training_methods --lr $learning_rate"

for lr in "${learning_rate[@]}"; do
  # 构建命令并执行
  cmd="$other_args 2>&1 | tee output_log/${training_methods}/trianing_lr${lr}_zostepsize${zo_step_size}_q_num${q_num}.log"
  echo "Running: $cmd"
  eval "$cmd"
  tmp+=1
done
