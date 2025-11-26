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
from models.pipeline_modeling_llama import PipelineScheduler, llama_stage
from llm_profile import stage_partition
import time
from lora import LoRA
from  torch.distributed import rpc
import torch.multiprocessing as mp
import os
from parse_args import parse_args
from distributed_model import DistributedSubModel

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

global vocab_size
global device

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
    
class AA:
    def __init__(self, name):
        self.name = name
        self.param = torch.randn(1000,128256, device=f"cuda:{name}")
        print(self.param)
        
    def hhh(self, num):
        print("{}: {}".format(self.name, num))
    
    def work0_work1(self, num):
        print("{}: {}".format(self.name, num))
        
    def master_work0(self, num):
        print("{}: {}".format(self.name, num))
        global_shared_rrefs["worker-1"].rpc_sync().work0_work1(11)
        
global global_shared_rrefs
global_shared_rrefs = {}

def register_rref(name, rref):
    global_shared_rrefs[name] = rref
    
    
def main(args):
    device = f"cuda:{args.gpus[0]}"
    
    model_split_configs, iszeroth_ls, device_to_stage_ls, stage_to_device_ls = stage_partition(args.model_name_or_path)
    remote_networks = {}
    for device_idx, (model_split_config, iszeroth, device_to_stage) in enumerate(zip(model_split_configs, iszeroth_ls, device_to_stage_ls)):
        remote_networks[f"worker-{device_idx}"] = rpc.remote(f"worker-{device_idx}", AA, args=(args.gpus[device_idx]))
    
    for other_worker in ["worker-0", "worker-1"]:
        for name, rref in remote_networks.items():
            rpc.rpc_sync(other_worker, register_rref, args=(name, rref))
            
    remote_networks[f"worker-0"].rpc_sync().master_work0(11)
    
    import time
    time.sleep(100)
    #train
    # for epoch in range(args.epochs):
    #     print('Epoch: {}, len(train_loaders): {}'.format(epoch, len(train_dataloader)))
    #     for iter_idx, (inputs, labels) in enumerate(train_dataloader):
    #         for stage_idx, device_idx in enumerate(stage_to_device_ls):
    #             if stage_idx == 0:
    #                 remote_networks[f"worker-{args.gpus[device_idx]}"].rpc_sync().train_iter(iter_idx, inputs=inputs)
    #             elif stage_idx == len(device_to_stage_ls)-1:
    #                 remote_networks[f"worker-{args.gpus[device_idx]}"].rpc_sync().train_iter(iter_idx, labels=labels)
    #             else:
    #                 remote_networks[f"worker-{args.gpus[device_idx]}"].rpc_sync().train_iter(iter_idx)

def init_process(rank, world_size, args):
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

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
    mp.spawn(init_process, args=(world_size, args), nprocs=world_size, join=True)
    
    
# def train_iter(self, epoch_idx, iter_idx, micro_iter_idx, inputs=None, labels=None):
#         self.optimizer.zero_grad()
#         self.network.train()
#         self.send_signal_ls = []
#         if self.network.is_zeroth:    #only forward
#             self.perturbs_dict_ls = []
#             self.params_dict = {
#                 name: p for name, p in self.network.named_parameters() if p.requires_grad
#                 }
#             with torch.no_grad():
#                 if self.network.include_embed:  #第一个stage
#                     self.base_inputs = to_device(inputs, self.device)
#                     self.f_theta = partial(self.f, inputs=self.base_inputs)
#                     base_outputs = self.f_theta(self.params_dict)
#                     self.send_signal_ls.append(self.send(base_outputs, self.next_device_name, "base_inputs"))
#                     self.call_method_by_name("start_"+self.args.method)
#                 elif self.network.include_lm_head:
#                     self.base_inputs = to_device(self.base_inputs, self.device)
#                     self.labels = to_device(labels, self.device)
#                     base_loss = self.f(self.params_dict, self.base_inputs, self.labels, ForCausalLMLoss)
#                     self.call_method_by_name("final_"+self.args.method, base_loss)   
#                 else:
#                     # self.f_theta = partial(self.f)
#                     self.base_inputs = to_device(self.base_inputs, self.device)
#                     base_outputs = self.f(self.params_dict, self.base_inputs)
#                     self.send_signal_ls.append(self.send(base_outputs, self.next_device_name, "base_inputs"))
#                     self.call_method_by_name(self.args.method)    
#             for send_signal in self.send_signal_ls:
#                 send_signal.wait()
#         else:   #forward and backward pass
#             if self.network.include_lm_head:  #最后一个stage
#                 # with torch.amp.autocast(device_type = self.device,enabled=self.args.amp):
#                 self.base_inputs = to_device(self.base_inputs, self.device)
#                 self.base_inputs['hidden_states'] = self.base_inputs['hidden_states'].detach().requires_grad_()
#                 self.labels = to_device(labels, self.device)
#                 logits = self.network(**self.base_inputs, use_cache=False)
#                 base_loss = self.criterion(logits=logits, labels=self.labels, vocab_size=self.args.vocab_size)
#                 base_loss.backward()
#                 if self.iszeroth_ls.count(True) + 1 < self.total_devices:
#                     # print("base_inputs.grad:", self.base_inputs['hidden_states'].grad)
#                     backward_signals = global_shared_rrefs[self.previous_device_name].rpc_async().loss_backward(self.base_inputs['hidden_states'].grad.cpu(), 2)
#                 base_loss_eval = base_loss.detach().cpu().item()
#                 print("base_loss: ", base_loss_eval)
#                 self.loss_writer.add_scalar("base_loss", base_loss_eval, global_step=iter_idx)
#                 if iter_idx % 1000 == 0:
#                     self.loss_writer.flush()
#                 if (iter_idx == self.args.total_iterations - 1) and (epoch_idx == self.args.epochs - 1):
#                     self.loss_writer.close()
#                 # del logits
#                 # torch.cuda.empty_cache()
#                 if self.estimated_inputs_ls is not None:
#                     self.send_signal_ls = []
#                     estimated_loss_ls = []
#                     with torch.no_grad():
#                         for estimated_inputs in self.estimated_inputs_ls:
#                             estimated_inputs = to_device(estimated_inputs, self.device)
#                             estimated_logits = self.network(**estimated_inputs, use_cache=False)
#                             # logits = estimated_outputs.logits
#                             estimated_loss = self.criterion(logits=estimated_logits, labels=self.labels, vocab_size=self.args.vocab_size).detach().cpu().item()
#                             estimated_loss_ls.append(estimated_loss)
#                             # print("esitmated_loss: ", esitmated_loss)
#                             directional_derivative = (estimated_loss - base_loss_eval) / self.args.zo_step_size
#                             for device_idx in range(len(self.iszeroth_ls)):
#                                 if self.iszeroth_ls[device_idx]:
#                                     self.send_signal_ls.append(
#                                         global_shared_rrefs[f"worker-{self.args.gpus[device_idx]}"].rpc_async().update_zeroth_model(directional_derivative,)
#                                         )
#                         self.loss_writer.add_histogram("estimate_loss", torch.tensor(estimated_loss_ls), global_step=iter_idx)
#                     for send_signal in self.send_signal_ls:
#                         send_signal.wait()
#                 backward_signals.wait()
#                 # self.optimizer.step()
#             elif self.network.include_embed:
#                 self.base_inputs = to_device(inputs, self.device)
#                 # self.base_inputs['hidden_states'] = self.base_inputs['hidden_states'].detach().requires_grad_()
#                 self.base_outputs = self.network(**self.base_inputs, use_cache=False)
#                 self.send_signal_ls.append(self.send(self.base_outputs, self.next_device_name, "base_inputs"))
#                 for send_signal in self.send_signal_ls:
#                     send_signal.wait()
#             else:
#                 # with torch.amp.autocast(device_type = self.device,enabled=self.args.amp):
#                 self.send_signal_ls = []
#                 self.base_inputs =  to_device(self.base_inputs, self.device)
#                 # print("base_inputs: ", self.base_inputs)
#                 self.base_inputs['hidden_states'] = self.base_inputs['hidden_states'].detach().requires_grad_()
#                 self.base_outputs = self.network(**self.base_inputs, use_cache=False)
#                 self.send_signal_ls.append(self.send(self.base_outputs, self.next_device_name, "base_inputs"))
#                 if self.estimated_inputs_ls is not None:
#                     with torch.no_grad():
#                         for q_cnt, estimated_inputs in enumerate(self.estimated_inputs_ls):
#                             estimated_inputs = to_device(estimated_inputs, self.device)
#                             estimated_outputs = self.network(**estimated_inputs, use_cache=False)
#                             self.send_signal_ls.append(self.send(estimated_outputs, self.next_device_name, "estimated_inputs_ls"+str(q_cnt)))
#                 for send_signal in self.send_signal_ls:
#                     send_signal.wait()
#                 # self.optimizer.step()


    def generate_subnetwork(self):
        #split model
        # with torch.device("meta"):
        model_config = AutoConfig.from_pretrained(self.args.model_name_or_path)
        # self.full_network = AutoModelForCausalLM.from_pretrained(
        #     self.args.model_name_or_path,
        #     from_tf=bool(".ckpt" in self.args.model_name_or_path),
        #     config=model_config,
        #     ignore_mismatched_sizes=True,
        #     torch_dtype=torch.bfloat16,
        #     attn_implementation="eager"
        # )
        self.network = llama_stage(base_model=self.full_network, 
                                    layer_start=self.model_split_config[0], 
                                    layer_end=self.model_split_config[1],
                                    include_embed=self.include_embed,
                                    include_lm_head=self.include_lm_head,
                                    is_zeroth=self.iszeroth,
                                    last_zeroth_model=True if self.iszeroth and not self.iszeroth_ls[self.next_device_idx] else False)
        del self.full_network
        torch.cuda.empty_cache()
        
        # if self.network.include_lm_head:
        # for name, param in self.network.named_parameters():
        #     if "embed" in name:
        #         print(param.data.equal(self.full_network.state_dict()["model.embed_tokens.weight"].data))
        #     elif "layers" in name:
        #         # print(name)
        #         name_split = name.split(".")
        #         name_split[1] = str(int(name_split[1])+self.model_split_config[0])
        #         tmp_name = ".".join(name_split)
        #         # print(tmp_name)
        #         print(param.data.equal(self.full_network.state_dict()["model."+tmp_name].data))
        #         # model.layers.0.self_attn.q_proj.weight
        #         # layers.0.self_attn.q_proj.weight
        # import sys
        # sys.exit(0)
        # self.network = self.network.to_empty(device="cpu")
        
        # if self.device_to_stage == 0:
        #     embe_token = torch.load("./model_part_layers/llama3b/embedding.pt")#.to(device)
        #     embe_token = {".".join(k.split(".")[1:]): v for k, v in embe_token.items()}
        #     self.network.load_state_dict(embe_token, strict=False, assign=True)
        
        # for layer_idx in range(self.model_split_config[0], self.model_split_config[1]):
        #     layer_tensors = torch.load("./model_part_layers/llama3b/layer_"+str(layer_idx)+".pt")#.to(device)
        #     layer_tensors  = {"layers."+str(layer_idx-self.model_split_config[0])+"."+k: v for k, v in layer_tensors.items()}
        #     self.network.load_state_dict(layer_tensors, strict=False, assign=True)
        
        # if self.device_to_stage == self.total_devices-1:
        #     head_norm = torch.load("./model_part_layers/llama3b/head_norm.pt")#.to(device)
        #     head_norm = {"norm.weight": v for k, v in head_norm.items()}
        #     self.network.load_state_dict(head_norm, strict=False, assign=True)
        #     embe_token = torch.load("./model_part_layers/llama3b/embedding.pt")#.to(device)
        #     embe_token = {"lm_head.weight": v for k, v in embe_token.items()}
        #     self.network.load_state_dict(embe_token, strict=False, assign=True)
        
        #load lora config
        replace_linear_with_lora(self.network, self.args.lora_r, self.args.lora_alpha)
        for name,param in self.network.named_parameters():
            if "lora" not in name:
                param.requires_grad = False
        # for name, param in self.network.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)
        # target_modules = ["q_proj","v_proj"]   #,"gate_proj","down_proj","up_proj"
        # lora_config = LoraConfig(
        #     r=self.args.lora_r,
        #     lora_alpha=self.args.lora_alpha,
        #     target_modules=target_modules,
        #     fan_in_fan_out=False,
        #     lora_dropout=0.1,
        #     inference_mode=False,
        #     bias="none",
        #     task_type="CAUSAL_LM"
        # )
        # # LoRA(sub_network_ls[subnetwork_idx], r=args.lora_r, alpha=args.lora_alpha, float16=False)
        # self.network = get_peft_model(self.network, lora_config)
        # # 将 LoRA 层权重转换为 bfloat16 格式
        # for name, param in self.network.named_parameters():
        #     # if "lora" in name:
        #         # param.data = param.data.to(torch.bfloat16)
        #         # print(name, param.data)
        #     if 'lora_A' in name:  # 只转换 LoRA 层的权重
        #         torch.nn.init.kaiming_uniform_(param.data, a=math.sqrt(5))
        #     # if "lora" in name:
        #     #     # param.data = param.data.to(torch.bfloat16)
        #     #     print(name, param.dtype)
        self.network = self.network.to(self.device)
        # self.network.print_trainable_parameters()



for micro_iter_idx, (inputs, labels) in enumerate(zip(micro_batch_datas_inputs, micro_batch_datas_labels)):
                # send labels to stage[-1] and last_zerothmodel
                receive_labels_signals = []
                receive_labels_signals.append(
                    remote_networks[f"worker-{args.gpus[stage_to_device_ls[-1]]}"].rpc_async().receive_labels(labels, micro_iter_idx)
                    )
                receive_labels_signals.append(
                    remote_networks[f"worker-{args.gpus[last_zeroth_submodel]}"].rpc_async().receive_labels(labels, micro_iter_idx)
                    )
                # base forward
                micro_base_forward_signals.append(
                    remote_networks[f"worker-{args.gpus[stage_to_device_ls[0]]}"].rpc_async().base_forward_only(epoch_idx, iter_idx, micro_iter_idx, inputs)
                )
            for receive_labels_signal in receive_labels_signals:
                receive_labels_signal.wait()
            #estimate forward
            micro_estimate_forward_signals = []
            for micro_iter_idx, (inputs, labels) in enumerate(zip(micro_batch_datas_inputs, micro_batch_datas_labels)):
                for q_cnt in range(args.q_num):
                    if q_cnt == 0:
                        micro_estimate_forward_signals.append(
                            remote_networks[f"worker-{args.gpus[stage_to_device_ls[0]]}"].rpc_async().estimate_grads(epoch_idx, 
                                                                                                                     iter_idx, micro_iter_idx, q_cnt, inputs=inputs)
                        )
                    else:
                        micro_estimate_forward_signals.append(
                            remote_networks[f"worker-{args.gpus[stage_to_device_ls[0]]}"].rpc_async().estimate_grads(epoch_idx, iter_idx, micro_iter_idx, q_cnt)
                        )
            #base backward
            for micro_iter_idx, micro_base_forward_signal in enumerate(micro_base_forward_signals):
                micro_base_forward_signal.wait()
                # wait_base_forward_signals = []
                # for stage_idx, device_idx in enumerate(stage_to_device_ls[:-1]):
                #     wait_base_forward_signals.append(
                #         remote_networks[f"worker-{args.gpus[device_idx]}"].rpc_async().wait_base_forward_signal(micro_iter_idx)  #DistributedSubModel.set_train_configuration
                #     )
                # for wait_base_forward_signal in wait_base_forward_signals:
                #     wait_base_forward_signal.wait()
                micro_base_backward_signals.append(
                    remote_networks[f"worker-{args.gpus[stage_to_device_ls[-1]]}"].rpc_async().base_backward_only(epoch_idx, iter_idx, micro_iter_idx)
                )
            #wait compute
            for micro_base_backward_signal in micro_base_backward_signals:
                micro_base_backward_signal.wait()
            for micro_estimate_forward_signal in micro_estimate_forward_signals:
                micro_estimate_forward_signal.wait()
                
                
                  # def _sched(deivce_name, epoch_idx, iter_idx, forward_mb_num):
                    #     base_forward_futs[forward_mb_num][stage_idx-1].wait()
                    #     base_forward_futs[forward_mb_num][stage_idx] = \
                    #         remote_networks[f"worker-{deivce_name}"].rpc_async().base_forward_only(epoch_idx, iter_idx, forward_mb_num)
                    # deivce_name = args.gpus[stage_to_device_ls[stage_idx]]
                    # threading.Thread(target=_sched, args=(deivce_name, epoch_idx, iter_idx, forward_mb_num)).start()
                    # deivce_name = args.gpus[stage_to_device_ls[stage_idx]]
                    # base_forward_futs[forward_mb_num][stage_idx] = base_forward_futs[forward_mb_num][stage_idx-1].then(
                    #     make_forward_callback(epoch_idx, iter_idx, forward_mb_num, deivce_name)
                    # )
                    # deivce_name = args.gpus[stage_to_device_ls[stage_idx]]
                    # base_forward_futs[forward_mb_num][stage_idx] = base_forward_futs[forward_mb_num][stage_idx-1].then(
                    #     lambda _, deivce_name=deivce_name: remote_networks[f"worker-{deivce_name}"].rpc_async().base_forward_only(epoch_idx, iter_idx, forward_mb_num)
                    # )
                    
                    

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
                                    F[mb][s] = remote_networks[W(s)].rpc_async().base_forward_only(epoch_idx, iter_idx, mb, None, (W(s-1), mb))
                                    f_started[mb][s] = True
                                    progressed = True

                    # ---------- backward ----------
                    for mb in range(M):
                        # 尾段反向：等最后一段前向完成即可
                        if not b_started[mb][0]:
                            last_f = F[mb][S-1]
                            if last_f is not None and last_f.done():
                                B[mb][0] = remote_networks[W(S-1)].rpc_async().base_backward_only(mb)
                                b_started[mb][0] = True
                                progressed = True

                        # 其余反向：等“下一段反向”完成
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
                    # if not progressed:
                    #     time.sleep(0.002)  # 2ms，按需调整
            elif args.training_methods == "only_first_backup":   #gpipe
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
                                base_forward_futs[forward_mb_num][0] = remote_networks[f"worker-{args.ip_addrs[stage_to_device_ls[0]]}"].rpc_async().base_forward_only(epoch_idx, 
                                                                                                                                iter_idx, forward_mb_num, 
                                                                                                                                micro_batch_datas_inputs[forward_mb_num])
                            else:
                                base_forward_futs[forward_mb_num][stage_idx-1].wait()
                                base_forward_futs[forward_mb_num][stage_idx] = \
                                    remote_networks[f"worker-{args.ip_addrs[stage_to_device_ls[stage_idx]]}"].rpc_async().base_forward_only(epoch_idx, iter_idx, forward_mb_num)
                    else:  #backward
                        for stage_idx, device_idx in enumerate(stage_to_device_ls):
                            inverse_stage = len(stage_to_device_ls) - 1 - stage_idx
                            backward_mb_num = step_idx - base_forward_total_steps - inverse_stage
                            if 0 <= backward_mb_num < args.micro_batch_num:
                                if inverse_stage == 0:
                                    base_forward_futs[backward_mb_num][len(stage_to_device_ls)-1].wait()
                                    base_backward_futs[backward_mb_num][inverse_stage] = \
                                        remote_networks[f"worker-{args.ip_addrs[stage_to_device_ls[stage_idx]]}"].rpc_async().base_backward_only(backward_mb_num)
                                else:
                                    base_backward_futs[backward_mb_num][inverse_stage-1].wait()
                                    # print("backward_mb_num: ", backward_mb_num, "inverse_stage: ", inverse_stage, "waiting")
                                    base_backward_futs[backward_mb_num][inverse_stage] = \
                                        remote_networks[f"worker-{args.ip_addrs[stage_to_device_ls[stage_idx]]}"].rpc_async().base_backward_only(backward_mb_num)
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
                                        remote_networks[f"worker-{args.ip_addrs[stage_to_device_ls[0]]}"].rpc_async().base_forward_only(epoch_idx,
                                                                                                                                    iter_idx, forward_mb_num,
                                                                                                                                    micro_batch_datas_inputs[forward_mb_num])
                                else:
                                    base_forward_futs[forward_mb_num][stage_idx-1].wait()
                                    base_forward_futs[forward_mb_num][stage_idx] = \
                                        remote_networks[f"worker-{args.ip_addrs[stage_to_device_ls[stage_idx]]}"].rpc_async().base_forward_only(epoch_idx, iter_idx, forward_mb_num)
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
                                    print("fowarad satge:{}, micro-batch {}".format(stage_idx, forward_mb_num))
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
                                    print("fowarad satge:{}, micro-batch {}".format(stage_idx, forward_mb_num))
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
                                    print("backward satge:{}, micro-batch {}".format(stage_idx, backward_mb_num))
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
                            print("forward satge:{}, micro-batch {} estimate_q_cnt {}".format(stage_idx, estimate_mb_num, estimate_q_cnt))
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
                    print("-"*50)
                for mb_num in range(args.micro_batch_num):
                    base_backward_futs[mb_num][iszeroth_ls.count(False)-1].wait()
                for mb_num in range(args.micro_batch_num):
                    for q_cnt in range(args.q_num):
                        estimate_forward_futs[mb_num][q_cnt][iszeroth_ls.count(True)-1].wait()

                remote_networks[f"worker-{args.ip_addrs[stage_to_device_ls[-1]]}"].rpc_sync().send_base_loss()
            