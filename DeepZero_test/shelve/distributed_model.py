import torch
import sys
from functools import reduce
from tools import *
from models.pipeline_modeling_llama import llama_stage
# from peft import (
#     LoraConfig,
#     PeftModel,
#     get_peft_model,
#     get_peft_model_state_dict,
#     prepare_model_for_kbit_training,
#     set_peft_model_state_dict,
# )
from data.math_dataloader import load_hf_tokenizer, smart_tokenizer_and_embedding_resize,make_supervised_data_module
import time
from functools import partial
from loss_utils import ForCausalLMLoss, fixed_cross_entropy, ForCausalLMLoss_chunked
from  torch.distributed import rpc
from types import MethodType
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed
)
from torch.cuda.amp import autocast
import math
# import torch.distributed.pipelining
# from torch.utils.tensorboard import SummaryWriter
from numpy import mean
from transformers.optimization import get_scheduler
import threading
from torch.nn.utils.stateless import functional_call
from torch.futures import Future as PyFuture

import re
from typing import Optional, Sequence, Union
import torch
from safetensors.torch import safe_open, save_file
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

sys.path.append(".")

global global_shared_rrefs
global_shared_rrefs = {}

def extract_llama_subset_safetensors(
    ckpt_path: str,
    keep_layers: Union[Sequence[int], range, tuple],   # 如 (start, end) 或 [8,9,10]
    include_embed: bool = True,
    include_norm: bool = True,
    include_lm_head: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    out_path: Optional[str] = None,
):
    if isinstance(keep_layers, tuple) and len(keep_layers) == 2:
        start, end = keep_layers
        keep_list = list(range(int(start), int(end)))
    elif isinstance(keep_layers, range):
        keep_list = list(keep_layers)
    else:
        keep_list = sorted({int(i) for i in keep_layers})
    remap = {old: new for new, old in enumerate(keep_list)}

    subset_sd = {}

    def _maybe_cast_to_dtype(t: torch.Tensor) -> torch.Tensor:
        return t.to(dtype) if torch.is_floating_point(t) else t

    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            # print("key:", k)
            # 2.1 头/尾：embed, norm, lm_head
            if k == "model.embed_tokens.weight":
                if include_embed:
                    subset_sd["embed_tokens.weight"] = _maybe_cast_to_dtype(f.get_tensor(k))
                elif include_lm_head:
                    subset_sd["lm_head.weight"] = _maybe_cast_to_dtype(f.get_tensor(k))
                continue
            if include_norm and k == "model.norm.weight":
                subset_sd["norm.weight"] = _maybe_cast_to_dtype(f.get_tensor(k))
                continue

            # 2.2 中间层：model.layers.{i}.<suffix>  ->  layers.{new_i}.<suffix>
            m = re.match(r"model\.layers\.(\d+)\.(.+)", k)
            if m:
                old_idx = int(m.group(1))
                if old_idx in remap:
                    new_idx = remap[old_idx]
                    suffix = m.group(2)
                    new_k = f"layers.{new_idx}.{suffix}"
                    subset_sd[new_k] = _maybe_cast_to_dtype(f.get_tensor(k))

    # 3) 可选：保存为新的 safetensors（仅包含子集）
    if out_path is not None:
        save_file(subset_sd, out_path)

    return subset_sd


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.empty(in_dim, rank).to(torch.bfloat16))
        torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # similar to standard weight initialization
        self.lora_B = torch.nn.Parameter(torch.zeros(rank, out_dim).to(torch.bfloat16))
        # torch.nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5)) 
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.lora_A @ self.lora_B)
        # print("lora output", torch.sum(torch.abs(x)))
        return x

class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)
    
def replace_linear_with_lora(model, lora_r, lora_alpha):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear) and ("q_proj" in name or "v_proj" in name):
            # Replace the Linear layer with LinearWithLoRA
            setattr(model, name, LinearWithLoRA(module, lora_r, lora_alpha))
        else:
            # Recursively apply the same function to child modules
            replace_linear_with_lora(module, lora_r, lora_alpha) 

def register_rref(name, rref):
    global_shared_rrefs[name] = rref
    
def to_device(inputs, device):
    if isinstance(inputs, dict):
        output = {}
        for k, v in inputs.items():
            try:
                output[k] = to_device(v, device)
            except:
                output[k] = v
    elif isinstance(inputs, list):
        output = []
        for v in inputs:
            try:
                output.append(to_device(v, device))
            except:
                output.append(v)
    elif isinstance(inputs, tuple):
        output = []
        for v in inputs:
            try:
                output.append(to_device(v, device))
            except:
                output.append(v)
        output = tuple(output)
    else:
        output = inputs.to(device)
    return output

def to_cpu(inputs):
    if isinstance(inputs, dict):
        output = {}
        for k, v in inputs.items():
            try:
                output[k] = to_cpu(v)
            except:
                output[k] = v
    elif isinstance(inputs, list):
        output = []
        for v in inputs:
            try:
                output.append(to_cpu(v))
            except:
                output.append(v)
    elif isinstance(inputs, tuple):
        output = []
        for v in inputs:
            try:
                output.append(to_cpu(v))
            except:
                output.append(v)
        output = tuple(output)
    else:
        output = None
        output = inputs.cpu()
    return output

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

class DistributedSubModel(object):
    def __init__(self, args, device_idx, model_split_config, iszeroth, device_to_stage, device_to_stage_ls, iszeroth_ls) -> None:
        super().__init__()
        self.args = args
        # self.full_network = full_network
        self.device_idx = device_idx
        self.device = "cuda"
        self.network = None
        self.model_split_config = model_split_config
        self.iszeroth = iszeroth
        self.device_to_stage = device_to_stage
        self.total_devices = len(device_to_stage_ls)
        self.include_embed = True if self.device_to_stage == 0 else False
        self.include_lm_head = True if self.device_to_stage == self.total_devices-1 else False
        self.iszeroth_ls = iszeroth_ls
        self.previous_devcie, self.next_device = None, None
        # if device_to_stage - 1 >= 0:
        self.previous_device_idx = (device_to_stage - 1) % len(device_to_stage_ls)
        # self.previous_device = f"cuda:{args.gpus[device_to_stage_ls.index(self.previous_device_idx)]}"
        self.previous_device_name = f"worker-{args.ip_addrs[device_to_stage_ls.index(self.previous_device_idx)]}"
        # if device_to_stage + 1 <= len(device_to_stage_ls):
        self.next_device_idx = (device_to_stage + 1) % len(device_to_stage_ls)
        # self.next_device = f"cuda:{args.gpus[device_to_stage_ls.index(self.next_device_idx)]}"
        self.next_device_name = f"worker-{args.ip_addrs[device_to_stage_ls.index(self.next_device_idx)]}"
        # print(self.device, self.previous_device_name, self.next_device_name)
        self.last_zeroth_model = True if self.iszeroth and (not self.iszeroth_ls[self.next_device_idx]) else False
                
        self.estimated_inputs_ls = [None for _ in range(self.args.q_num)]
        self.params_dict = None
        self.f_theta = None
        self.q_cnt = 0
        # network1 = 
        self.generate_subnetwork()
        # self._build_model()
        # network2 = self._build_model()
        # model_stat1 = network1.state_dict()
        # model_stat2 = network2.state_dict()
        # for name, param in model_stat1.items():
        #     print(param.data.dtype)
        #     tmp = param==model_stat2[name]
        #     print(tmp.all())
        
        if self.network.include_lm_head:
            args.board_output_dir = "lr"+str(args.lr)+"zo_step_size"+str(args.zo_step_size)+"q_num"+str(args.q_num)
            # self.loss_writer = SummaryWriter(log_dir="lossboard/"+args.board_output_dir)
            # self.esitmateloss_writer = SummaryWriter(log_dir="output_log/estimate_loss")
        self.optimizer = torch.optim.Adam(
            (p for p in self.network.parameters() if p.requires_grad),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )
        self.lr_scheduler = get_scheduler(
                "cosine",
                optimizer=self.optimizer,
                num_warmup_steps=int(0.1*args.epoch*args.total_iterations),
                num_training_steps=int(args.epoch*args.total_iterations)
            )
        self.criterion = ForCausalLMLoss
        # global_length = (args.epoch - args.warm_epochs) * len(train_data)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=global_length)
        print("model loaded....")
        self.watch_video_memory()
    
    def watch_video_memory(self):
        allocated_memory = torch.cuda.memory_allocated(self.device)  # 已分配的显存
        reserved_memory = torch.cuda.memory_reserved(self.device)    # 已保留的显存
        print("{}  - 已分配的显存: {:.2f} GB".format(self.device, allocated_memory / (1024 ** 3)))
        print("{}  - 已保留的显存: {:.2f} GB".format(self.device, reserved_memory / (1024 ** 3)))
    
    def _build_model(self):
        self.args.vocab_size = 128258
        cfg = AutoConfig.from_pretrained(self.args.model_name_or_path)
        with torch.device("meta"):
            full = AutoModelForCausalLM.from_pretrained(
                self.args.model_name_or_path,
                config=cfg,
                ignore_mismatched_sizes=True,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )#.to("cpu")

        self.network = llama_stage(
            base_model=full,
            layer_start=self.model_split_config[0], 
            layer_end=self.model_split_config[1],
            include_embed=self.include_embed,
            include_lm_head=self.include_lm_head,
            is_zeroth=self.iszeroth,
            last_zeroth_model=self.last_zeroth_model  # 这里不在子模块标记“最后 zeroth”，以免和通信耦合
        )
        # print(network1)
        del full
        torch.cuda.empty_cache()
        self.network = self.network.to_empty(device="cpu")
        self.network.rotary_emb = LlamaRotaryEmbedding(cfg)
        sd = extract_llama_subset_safetensors(
            ckpt_path=f"{self.args.model_name_or_path}/model.safetensors",
            keep_layers=(self.model_split_config[0], self.model_split_config[1]),
            include_embed=self.include_embed,
            include_norm=True if self.include_lm_head or self.last_zeroth_model else False,
            include_lm_head=True if self.include_lm_head or self.last_zeroth_model else False,
            dtype=torch.bfloat16,
            out_path=None,
        )
        # print("sd: ", sd)
        self.network.load_state_dict(sd)
        # for name, params in network1.named_parameters():
        #     print(name, params)
        # return network1
        # LoRA 仅训练 LoRA 参数
        replace_linear_with_lora(self.network, self.args.lora_r, self.args.lora_alpha)
        for n, p in self.network.named_parameters():
            if "lora" not in n:
                p.requires_grad = False
        self.network = self.network.to(self.device)


    def generate_subnetwork(self):
        model_config = AutoConfig.from_pretrained(self.args.model_name_or_path)
        self.full_network = AutoModelForCausalLM.from_pretrained(
            self.args.model_name_or_path,
            config=model_config,
            ignore_mismatched_sizes=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            # attn_implementation="eager"
        )
        self.full_network = self.full_network.to("cpu")
        
        # print(self.device, self.last_zeroth_model)
        self.network = llama_stage(base_model=self.full_network, 
                                    layer_start=self.model_split_config[0], 
                                    layer_end=self.model_split_config[1],
                                    include_embed=self.include_embed,
                                    include_lm_head=self.include_lm_head,
                                    is_zeroth=self.iszeroth,
                                    last_zeroth_model=self.last_zeroth_model)
        del self.full_network
        torch.cuda.empty_cache()
        self.args.vocab_size = 128258
        #load lora config
        # for name,param in self.network.named_parameters():
        #     print(name, param)
        # return self.network
        replace_linear_with_lora(self.network, self.args.lora_r, self.args.lora_alpha)
        for name,param in self.network.named_parameters():
            if "lora" not in name:
                param.requires_grad = False
        self.network = self.network.to(self.device)
        # self.network.print_trainable_parameters()

    def set_supervised_data_module(self, tokenizer):
        self.train_dataset, self.data_collator, self.train_dataloader = make_supervised_data_module(tokenizer, self.args)
            
    @torch.no_grad()
    def f(self, params_dict, inputs, labels=None):
        # with torch.amp.autocast(device_type = self.device,enabled=self.args.amp):
        state_dict_backup = {
            name: param.clone() for name, param in self.network.named_parameters() if param.requires_grad
        }
        self.network.load_state_dict(params_dict, strict=False)
        if self.network.last_zeroth_model:
            output, logits = self.network(**inputs, use_cache=False)
        else:
            output = self.network(**inputs, use_cache=False)
        self.network.load_state_dict(state_dict_backup, strict=False)
        # del state_dict_backup
        # torch.cuda.empty_cache()
        if labels is not None:
            loss = self.criterion(logits=logits, labels=labels, vocab_size=self.args.vocab_size)
            del logits
            torch.cuda.empty_cache()
            return output, loss
        else:
            return output

    def send(self, send_value, to_device_name, attr_name):
        send_signal = global_shared_rrefs[to_device_name].rpc_async().receive(to_cpu(send_value), attr_name)
        # rpc.rpc_async(to_device, self.receive, args=(to_device(send_value, "cpu"), attr_name))
        return send_signal
    
    def base_forward_receive(self, micro_iter_idx, tensor):
        if self.iszeroth:
            self.base_inputs = to_device(tensor, self.device)
        else:
            self.base_inputs[micro_iter_idx] = to_device(tensor, self.device)
    
    def base_backward_receive(self, micro_iter_idx, tensor):
        self.output_gradients = to_device(tensor, self.device)
        
    def estimate_grads_receive(self, micro_iter_idx, q_cnt, tensor):
        self.estimated_inputs[micro_iter_idx][q_cnt] = to_device(tensor, self.device)
            
    def receive(self, micro_iter_idx, tensor, attr_name):
        # if hasattr(self, attr_name):
        if attr_name.startswith("estimated_inputs_ls"):
            # print("attr_name: ", attr_name)
            # print("num: ", int(attr_name[len("estimated_inputs_ls"):]))
            self.estimated_inputs_ls[int(attr_name[len("estimated_inputs_ls"):])] = tensor
        else:
            # print("attr_name: ", attr_name)
            setattr(self, attr_name, tensor)
    
    def broadcast(self, tensor, attr_name):
        send_signals = []
        for device_idx in range(len(self.iszeroth_ls)):
            if self.iszeroth_ls[device_idx]:
                send_signals.append(
                    global_shared_rrefs[f"worker-{self.args.ip_addrs[device_idx]}"].rpc_async().receive(to_device(tensor, "cpu"), attr_name)
                    )
        for send_signal in send_signals:
            send_signal.wait()
    
    def start_rge(self):
        # print("start rge")
        for q_cnt in range(self.args.q_num):
            perturbs_dict, perturbed_params_dict = {}, {}
            for key, param in self.params_dict.items():
                perturb = torch.randn_like(param)
                perturb /= (torch.norm(perturb) + 1e-8)
                # perturb = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                perturbs_dict[key] = perturb
                # print("perturbs_dict[{}]: {}".format(key, perturbs_dict[key]))
                perturbed_params_dict[key] = self.args.zo_step_size * perturb + param
            self.perturbs_dict_ls.append(perturbs_dict)
            estimated_outputs = self.f_theta(perturbed_params_dict)
            # print("send signal", q_cnt)
            self.estimated_signal_ls.append(self.send(estimated_outputs, self.next_device_name, "estimated_inputs_ls"+str(q_cnt)))
            # estimated_outputs_ls.append(estimated_outputs)
            
    def rge(self):
        for q_cnt, estimated_inputs in enumerate(self.estimated_inputs_ls):
            estimated_inputs = to_device(estimated_inputs, self.device)
            perturbs_dict, perturbed_params_dict = {}, {}
            for key, param in self.params_dict.items():
                perturb = torch.randn_like(param)
                perturb /= (torch.norm(perturb) + 1e-8)
                # perturb = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                perturbs_dict[key] = perturb
                perturbed_params_dict[key] = self.args.zo_step_size * perturb + param
            self.perturbs_dict_ls.append(perturbs_dict)
            estimated_outputs = self.f(perturbed_params_dict, estimated_inputs)
            self.send_signal_ls.append(self.send(estimated_outputs, self.next_device_name, "estimated_inputs_ls"+str(q_cnt)))
        
    def final_rge(self, base_loss):
        for q_cnt, estimated_inputs in enumerate(self.estimated_inputs_ls):
            estimated_inputs = to_device(estimated_inputs, self.device)
            perturbs_dict, perturbed_params_dict = {}, {}
            for key, param in self.params_dict.items():
                perturb = torch.randn_like(param)
                perturb /= (torch.norm(perturb) + 1e-8)
                # perturb = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                perturbs_dict[key] = perturb
                perturbed_params_dict[key] = self.args.zo_step_size * perturb + param
            self.perturbs_dict_ls.append(perturbs_dict)
            estimated_loss = self.f(perturbed_params_dict, estimated_inputs, self.labels, ForCausalLMLoss)
            directional_derivative = (estimated_loss - base_loss) / self.args.zo_step_size
            for device_idx in range(len(self.iszeroth_ls)-1):
                if self.iszeroth_ls[device_idx]:
                    self.send_signal_ls.append(
                        global_shared_rrefs[f"worker-{self.args.ip_addrs[device_idx]}"].rpc_async().update_zeroth_model(directional_derivative,)
                        )
            self.update_zeroth_model(directional_derivative)
        for send_signal in self.send_signal_ls:
            send_signal.wait()
    
    def call_method_by_name(self, method_name):
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            if callable(method):
                return method()
            else:
                raise ValueError(f"{method_name} is not callable.")
        else:
            raise AttributeError(f"{method_name} does not exist.")
    
    def set_train_configuration(self):
        self.optimizer.zero_grad()
        self.network.train()
        self.base_forward_signals = {}
        self.estimated_signal_ls = {}
        self.labels = {micro_batch_idx: None for micro_batch_idx in range(self.args.micro_batch_num)}
        self.base_loss = {micro_batch_idx: None for micro_batch_idx in range(self.args.micro_batch_num)}
        self.base_loss_eval = {}
        if self.network.is_zeroth:    #only forward
            self.perturbs_dict = {micro_batch_idx: {} for micro_batch_idx in range(self.args.micro_batch_num)}
            self.params_dict = {
                name: p for name, p in self.network.named_parameters() if p.requires_grad
                }
            self.base_inputs = None
            self.estimated_loss = {micro_batch_idx: {} for micro_batch_idx in range(self.args.micro_batch_num)}
            self.directional_derivative = {micro_batch_idx: {} for micro_batch_idx in range(self.args.micro_batch_num)}
            if self.network.include_embed:
                self.estimated_inputs = {micro_batch_idx: None for micro_batch_idx in range(self.args.micro_batch_num)}
            else:
                self.estimated_inputs = {micro_batch_idx: {} for micro_batch_idx in range(self.args.micro_batch_num)}
        else:
            if self.network.include_embed:
                self.base_inputs = None
            else:
                self.base_inputs = {}
            self.base_outputs = {}
        if self.network.include_lm_head:
            self.directional_derivative_ls = [] 
            self.estimated_loss_ls = []
                   
    def receive_labels(self, labels, micro_iter_idx):
        self.labels[micro_iter_idx] = to_device(labels, self.device)
    
    def wait_base_forward_signal(self, micro_iter_idx):
        self.base_forward_signals[micro_iter_idx].wait()
        
    def base_forward_only(self, epoch_idx, iter_idx, micro_iter_idx, inputs=None):
        if self.network.is_zeroth:
            with torch.no_grad():
                if self.network.include_embed:
                    self.base_inputs = to_device(inputs, self.device)
                if self.network.last_zeroth_model:
                    base_outputs, intermediate_loss = self.f(self.params_dict, self.base_inputs, self.labels[micro_iter_idx])
                else:
                    base_outputs = self.f(self.params_dict, self.base_inputs, self.labels[micro_iter_idx])
                
                if self.network.include_lm_head:
                    self.base_loss_eval[micro_iter_idx] = base_outputs.detach().cpu().item()
                elif self.network.last_zeroth_model:
                    self.base_loss_eval[micro_iter_idx] = intermediate_loss.detach().cpu().item()
                    global_shared_rrefs[self.next_device_name].rpc_sync().base_forward_receive(micro_iter_idx, to_cpu(base_outputs))
                else:
                    global_shared_rrefs[self.next_device_name].rpc_sync().base_forward_receive(micro_iter_idx, to_cpu(base_outputs))              
        else:  #forward and backward pass
            if self.network.include_lm_head:  #最后一个stage
                self.base_inputs[micro_iter_idx]['hidden_states'] = self.base_inputs[micro_iter_idx]['hidden_states'].detach().requires_grad_()
                logits = self.network(**self.base_inputs[micro_iter_idx], use_cache=False)
                self.base_loss[micro_iter_idx] = self.criterion(logits=logits, labels=self.labels[micro_iter_idx], 
                                                                vocab_size=self.args.vocab_size) / self.args.micro_batch_num
                del logits
                torch.cuda.empty_cache()
                self.base_loss_eval[micro_iter_idx] = self.base_loss[micro_iter_idx].detach().cpu().item()
                if micro_iter_idx == self.args.micro_batch_num - 1:
                    mean_loss_value = sum(list(self.base_loss_eval.values()))
                    print("mean_loss_value: ", mean_loss_value)
                    # self.loss_writer.add_scalar("base_loss", mean_loss_value, global_step=iter_idx)
                    # if iter_idx % 1000 == 0:
                    #     self.loss_writer.flush()
                    # if (iter_idx == self.args.total_iterations - 1) and (epoch_idx == self.args.epochs - 1):
                    #     self.loss_writer.close()
            elif self.network.include_embed:
                self.base_inputs = to_device(inputs, self.device)
                self.base_outputs[micro_iter_idx] = self.network(**self.base_inputs, use_cache=False)
                global_shared_rrefs[self.next_device_name].rpc_sync().base_forward_receive(micro_iter_idx, to_cpu(self.base_outputs[micro_iter_idx]))
            else:
                self.base_inputs[micro_iter_idx]['hidden_states'] = self.base_inputs[micro_iter_idx]['hidden_states'].detach().requires_grad_()
                self.base_outputs[micro_iter_idx] = self.network(**self.base_inputs[micro_iter_idx], use_cache=False)
                global_shared_rrefs[self.next_device_name].rpc_sync().base_forward_receive(micro_iter_idx, to_cpu(self.base_outputs[micro_iter_idx]))
    
    def loss_backward(self, micro_iter_idx, gradient, backward_cnt):
        gradient = gradient.to(self.device)
        self.base_outputs['hidden_states'].backward(gradient)
        if self.iszeroth_ls.count(True) + backward_cnt < self.total_devices:
            backward_signals = global_shared_rrefs[self.previous_device_name].rpc_async().loss_backward(micro_iter_idx, 
                                                                                                        self.base_inputs[micro_iter_idx]['hidden_states'].grad.cpu(), 
                                                                                                        backward_cnt+1)
            backward_signals.wait()
        # self.optimizer.step()
        
    def base_backward_only(self, micro_iter_idx):
        if self.network.include_lm_head:
            self.base_loss[micro_iter_idx].backward()
            del self.base_loss[micro_iter_idx]
        else:
            self.base_outputs[micro_iter_idx]['hidden_states'].backward(self.output_gradients)
            del self.base_outputs[micro_iter_idx]
        
        if ((not self.iszeroth) and (not self.iszeroth_ls[self.previous_device_idx])) and self.device_idx != 0:
            tmp_send_signals = \
                global_shared_rrefs[self.previous_device_name].rpc_async().base_backward_receive(micro_iter_idx, 
                                                                                                self.base_inputs[micro_iter_idx]['hidden_states'].grad.cpu())
            del self.base_inputs[micro_iter_idx]
            tmp_send_signals.wait()
        torch.cuda.empty_cache()
    
    def estimate_grads(self, epoch_idx, iter_idx, micro_iter_idx, q_cnt, inputs=None):  
        if self.network.is_zeroth:    #only forward
            with torch.no_grad():
                if self.network.include_embed:  #第一个stage
                    if q_cnt == 0:
                        self.estimated_inputs[micro_iter_idx] = to_device(inputs, self.device)
                    # print("start rge")
                    micro_q_perturbs_dict, micro_q_perturbed_params_dict = {}, {}
                    for key, param in self.params_dict.items():
                        perturb = torch.randn_like(param)
                        perturb /= (torch.norm(perturb) + 1e-8)
                        # perturb = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                        micro_q_perturbs_dict[key] = perturb
                        micro_q_perturbed_params_dict[key] = self.args.zo_step_size * perturb + param
                    self.perturbs_dict[micro_iter_idx][q_cnt] = micro_q_perturbs_dict
                    estimated_outputs = self.f(micro_q_perturbed_params_dict, self.estimated_inputs[micro_iter_idx])
                    global_shared_rrefs[self.next_device_name].rpc_sync().estimate_grads_receive(micro_iter_idx, q_cnt, to_cpu(estimated_outputs))
                elif self.network.include_lm_head:
                    self.estimated_inputs[micro_iter_idx][q_cnt] = to_device(inputs, self.device)
                    micro_q_perturbs_dict, micro_q_perturbed_params_dict = {}, {}
                    for key, param in self.params_dict.items():
                        perturb = torch.randn_like(param)
                        perturb /= (torch.norm(perturb) + 1e-8)
                        # perturb = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                        micro_q_perturbs_dict[key] = perturb
                        micro_q_perturbed_params_dict[key] = self.args.zo_step_size * perturb + param
                    self.perturbs_dict_ls.append(micro_q_perturbs_dict)
                    micro_q_estimated_loss = self.f(micro_q_perturbed_params_dict, self.estimated_inputs[micro_iter_idx][q_cnt], self.labels)
                    micro_q_directional_derivative = (micro_q_estimated_loss - self.base_loss) / self.args.zo_step_size
                    self.directional_derivative_ls.append(micro_q_directional_derivative)
                elif self.last_zeroth_model and q_cnt >= 0:
                    # self.estimated_inputs = to_device(inputs, self.device)
                    micro_q_perturbs_dict, micro_q_perturbed_params_dict = {}, {}
                    for key, param in self.params_dict.items():
                        perturb = torch.randn_like(param)
                        perturb /= (torch.norm(perturb) + 1e-8)
                        # perturb = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                        micro_q_perturbs_dict[key] = perturb
                        micro_q_perturbed_params_dict[key] = self.args.zo_step_size * perturb + param
                    self.perturbs_dict[micro_iter_idx][q_cnt] = micro_q_perturbs_dict
                    _, micro_q_estimated_loss = self.f(micro_q_perturbed_params_dict, self.estimated_inputs[micro_iter_idx][q_cnt], self.labels[micro_iter_idx])
                    self.estimated_loss[micro_iter_idx][q_cnt] = micro_q_estimated_loss
                else:
                    # self.estimated_inputs = to_device(inputs, self.device)
                    micro_q_perturbs_dict, micro_q_perturbed_params_dict = {}, {}
                    for key, param in self.params_dict.items():
                        perturb = torch.randn_like(param)
                        perturb /= (torch.norm(perturb) + 1e-8)
                        # perturb = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                        micro_q_perturbs_dict[key] = perturb
                        micro_q_perturbed_params_dict[key] = self.args.zo_step_size * perturb + param
                    self.perturbs_dict[micro_iter_idx][q_cnt] = micro_q_perturbs_dict
                    estimated_outputs = self.f(micro_q_perturbed_params_dict, self.estimated_inputs[micro_iter_idx][q_cnt])
                    global_shared_rrefs[self.next_device_name].rpc_sync().estimate_grads_receive(micro_iter_idx, q_cnt, to_cpu(estimated_outputs))
        else:   #forward and backward pass
            with torch.no_grad():
                if self.network.include_lm_head:  #最后一个stage
                    # estimated_loss_ls = []
                    # self.estimated_inputs[micro_iter_idx][q_cnt] = to_device(inputs, self.device)
                    estimated_logits = self.network(**self.estimated_inputs[micro_iter_idx][q_cnt], use_cache=False)
                    # logits = estimated_outputs.logits
                    micro_q_estimated_loss = self.criterion(logits=estimated_logits, labels=self.labels, vocab_size=self.args.vocab_size).detach().cpu().item()
                    self.estimated_loss_ls.append(micro_q_estimated_loss)
                    # print("esitmated_loss: ", esitmated_loss)
                    micro_q_directional_derivative = (micro_q_estimated_loss - self.base_loss_eval) / self.args.zo_step_size
                    self.directional_derivative_ls.append(micro_q_directional_derivative)
                    # if q_cnt == self.args.q_num:
                    #     self.loss_writer.add_histogram("estimate_loss", torch.tensor(self.estimated_loss_ls), global_step=iter_idx)
                else:
                    # self.estimated_inputs[micro_iter_idx][q_cnt] = to_device(inputs, self.device)
                    estimated_outputs = self.network(**self.estimated_inputs[micro_iter_idx][q_cnt], use_cache=False)
                    self.estimated_signal_ls[q_cnt] = \
                        global_shared_rrefs[self.next_device_name].rpc_async().estimate_grads(epoch_idx, iter_idx, micro_iter_idx, q_cnt, to_cpu(estimated_outputs))
    
    def send_base_loss(self):
        for i in range(len(self.iszeroth_ls)):
            if not self.iszeroth_ls[i+1]:
                global_shared_rrefs[f"worker-{self.args.ip_addrs[i]}"].rpc_sync().zeroth_update(self.base_loss_eval)
                break
    
    def update_zeroth_model(self, directional_derivatives):
        grads_dict = {}
        # for perturbs_dict, directional_derivative in zip(self.perturbs_dict_ls, directional_derivative_ls):
        for micro_iter_idx in range(self.args.micro_batch_num):
            for q_cnt in range(self.args.q_num):
                if len(grads_dict.keys()) == len(self.params_dict.keys()):
                    for key, perturb in self.perturbs_dict[micro_iter_idx][self.q_cnt].items():
                        grads_dict[key] += perturb * directional_derivatives[micro_iter_idx][q_cnt] / self.args.q_num
                else:
                    for key, perturb in self.perturbs_dict[micro_iter_idx][self.q_cnt].items():
                        grads_dict[key] = perturb * directional_derivatives[micro_iter_idx][q_cnt] / self.args.q_num
        for key, param in self.params_dict.items():
            param.grad = grads_dict[key] / self.args.micro_batch_num
            # self.optimizer.step()
            
    def zeroth_update(self, base_loss_eval):
        # for q_cnt, directional_derivative in enumerate(self.directional_derivative_ls):
        for micro_iter_idx in range(self.args.micro_batch_num):
            for q_cnt in range(self.args.q_num):
                micro_q_directional_derivative = (self.estimated_loss[micro_iter_idx][q_cnt] - base_loss_eval[micro_iter_idx]) / self.args.zo_step_size
                self.directional_derivative[micro_iter_idx][q_cnt] = micro_q_directional_derivative
        zero_update_signals = []
        for device_idx in range(self.iszeroth_ls.count(True)-1):
            if self.iszeroth_ls[device_idx]:
                zero_update_signals.append(
                    global_shared_rrefs[f"worker-{self.args.ip_addrs[device_idx]}"].rpc_async().update_zeroth_model(to_cpu(self.directional_derivative))
                    )
        self.update_zeroth_model(self.directional_derivative)
        for zero_update_signal in zero_update_signals:
            zero_update_signal.wait()
            
    def network_optimzier(self):
        self.optimizer.step()
        self.lr_scheduler.step()
        
    def watch_gradient(self):
        for name,param in self.network.named_parameters():
            if param.requires_grad:
                avg_abs, mode_magnitude = tensor_abs_stats(param.grad)
                print("name: {}, avg_abs: {}, mode_magnitude: {}".format(name, avg_abs, mode_magnitude))
