import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser("Pretraining")
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--seed', type=int, default=123334)
    # parser.add_argument('--device', type=int, default=2)
    
    #model config
    parser.add_argument('--model', choices=['vit', 'llama', 'qwen'], default='llama')
    parser.add_argument('--model_name_or_path', type=str, default="../Llama-3.2-3B/")
    
    #data config
    parser.add_argument('--max_seq_len', type=int, default=64)
    parser.add_argument('--dataset', choices=['agnews','multirc', 'squad', 'math'], default='agnews')
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--micro_batch_size', type=int, default=4)
    parser.add_argument('--micro_batch_num', type=int, default=4)
    # parser.add_argument('--eval_batch_size', type=int, default=16)
    # parser.add_argument('--eval_micro_batch_size', type=int, default=4)
    parser.add_argument('--eval_micro_batch_num', type=int, default=4)
    parser.add_argument('--vocab_size', type=int, default=128256)
    
    #parallelism
    parser.add_argument('--training_methods', type=str, choices=['only_zeroth', 'only_first', 'hybrid', 'HDP', 'HPP', 'Mobius', 'pipedream'], default="hybrid")
    
    #comm
    parser.add_argument('--master-addr', type=str, default='192.168.1.239')
    parser.add_argument('--master-port', type=str, default='29516')
    parser.add_argument('--ip-addr', type=str, default='0')
    parser.add_argument('--ip-addrs', type=str, default='239,238,203,197,118,140')
    parser.add_argument('--ifname', type=str, default='eth0')

    # output
    parser.add_argument('--board_output_dir', type=str, default='base')
    
    #trianing args
    parser.add_argument('--cur_iter', type=int, default=0)
    parser.add_argument('--total_iterations', type=int, default=0)
    parser.add_argument('--zo_step_size', type=float, default=1e-3)
    
    #hyperparameter
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--q_num', type=int, default=3)
    parser.add_argument('--start_q_num', type=int, default=0)
    parser.add_argument('--epoch', type=int, default=4)
    
    args = parser.parse_args()
    return args