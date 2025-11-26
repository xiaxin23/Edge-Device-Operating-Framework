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

sys.path.append(".")

global global_shared_rrefs
global_shared_rrefs = {}

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

def watch_video_memory(device):
    allocated_memory = torch.cuda.memory_allocated(device)  # 已分配的显存
    reserved_memory = torch.cuda.memory_reserved(device)    # 已保留的显存
    print("{}  - 已分配的显存: {:.2f} GB".format(device, allocated_memory / (1024 ** 3)))
    print("{}  - 已保留的显存: {:.2f} GB".format(device, reserved_memory / (1024 ** 3)))
        
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
        
        self._mb_lock = threading.Lock()
        # —— 前向输出邮筒：{mb: Future(cpu_tensor or dict)}
        self._fwd_outbox_fut = {}
        # —— 反向梯度邮筒：{mb: Future(cpu_tensor)}
        self._bwd_inbox_fut = {}
        # —— 最后一段的 loss 邮筒：{mb: Future(loss_tensor)}
        self._loss_fut = {}
        # —— 幂等标记
        self._fwd_seen = set()    # 已执行过 forward 的 mb
        self._bwd_seen = set()    # 已执行过 backward 的 mb
        
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
                
        # self.estimated_inputs_ls = [None for _ in range(self.args.q_num)]
        self.params_dict = None
        self.f_theta = None
        self.q_cnt = 0
        self.load_model()
        self.generate_subnetwork()
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
        watch_video_memory(self.device)
    
    def load_model(self, ):
        #load tokenizer     
        # tokenizer = AutoTokenizer.from_pretrained(
        #     self.args.model_name_or_path,
        #     model_max_length=self.args.max_seq_len,
        #     padding_side="right",
        #     use_fast=False,
        # )

        # special_tokens_dict = dict()
        # if tokenizer.pad_token is None:
        #     special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        
        #load model    
        model_config = AutoConfig.from_pretrained(self.args.model_name_or_path)
        # with torch.device("meta"):
        self.full_network = AutoModelForCausalLM.from_pretrained(
            self.args.model_name_or_path,
            # from_tf=bool(".ckpt" in args.model_name_or_path),
            config=model_config,
            ignore_mismatched_sizes=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            # attn_implementation="eager"
        )
        self.full_network = self.full_network.to("cpu")
    
    def generate_subnetwork(self):
        #split model
        model_config = AutoConfig.from_pretrained(self.args.model_name_or_path)
        # print(self.device, self.last_zeroth_model)
        self.network = llama_stage(base_model=self.full_network, 
                                    layer_start=self.model_split_config[0], 
                                    layer_end=self.model_split_config[1],
                                    include_embed=self.include_embed,
                                    include_lm_head=self.include_lm_head,
                                    is_zeroth=self.iszeroth,
                                    last_zeroth_model=self.last_zeroth_model)
        del self.full_network
        self.args.vocab_size = 128258
        torch.cuda.empty_cache()
        #load lora config
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
        
        # base = dict(self.network.named_parameters(remove_duplicate=True))
        # base.update(params_dict)
        # output = functional_call(self.network, base, args=(), kwargs={**inputs, "use_cache": False})
        # if self.network.last_zeroth_model:
        #     output, logits = output
        
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
        fut = self._bwd_inbox_fut.get(micro_iter_idx)
        if fut is None:
            fut = torch.futures.Future()
            self._bwd_inbox_fut[micro_iter_idx] = fut
        if not fut.done():
            fut.set_result(to_device(tensor, self.device))  # 直接喂给等待中的反向
        # self.output_gradients[micro_iter_idx] = to_device(tensor, self.device)
        
    def estimate_grads_receive(self, micro_iter_idx, q_cnt, tensor):
        self.estimated_inputs[micro_iter_idx][q_cnt]= to_device(tensor, self.device)
            
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
    
    def _ensure_mailboxes(self):
        if not hasattr(self, "_fwd_outbox_fut"): self._fwd_outbox_fut = {}
        if not hasattr(self, "_bwd_inbox_fut"): self._bwd_inbox_fut = {}
        if not hasattr(self, "_loss_fut"):      self._loss_fut = {}
        if not hasattr(self, "_fwd_seen"):      self._fwd_seen = set()
        if not hasattr(self, "_bwd_seen"):      self._bwd_seen = set()
        
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
            self.output_gradients = {}
        if self.network.include_lm_head:
            self.directional_derivative_ls = [] 
            self.estimated_loss_ls = []
        
        self._ensure_mailboxes()
        self._fwd_outbox_fut.clear()
        self._bwd_inbox_fut.clear()
        self._loss_fut.clear()
        self._fwd_seen.clear()
        self._bwd_seen.clear()
                   
    def receive_labels(self, labels, micro_iter_idx):
        self.labels[micro_iter_idx] = to_device(labels, self.device)
    
    def wait_base_forward_signal(self, micro_iter_idx):
        self.base_forward_signals[micro_iter_idx].wait()
    
    # 由本段 forward 存放输出（供下一段拉取）
    def _stash_forward_out(self, micro_iter_idx, tensor):
        with self._mb_lock:
            fut = self._fwd_outbox_fut.get(micro_iter_idx)
            if fut is None:
                fut = torch.futures.Future()
                self._fwd_outbox_fut[micro_iter_idx] = fut
            if not fut.done():                      # 防止重复 set_result
                fut.set_result(to_cpu(tensor))

    def fetch_forward_out(self, micro_iter_idx):
        fut = self._fwd_outbox_fut.get(micro_iter_idx)
        if fut is None:
            fut = torch.futures.Future()
            self._fwd_outbox_fut[micro_iter_idx] = fut
        out = fut.wait()                        # 生产者没放时会阻塞等待
        self._fwd_outbox_fut.pop(micro_iter_idx, None)  # 取一次就清
        return out
    
    def _stash_loss(self, micro_iter_idx, loss_tensor):
        fut = self._loss_fut.get(micro_iter_idx)
        if fut is None:
            fut = torch.futures.Future()
            self._loss_fut[micro_iter_idx] = fut
        if not fut.done():
            fut.set_result(loss_tensor)

    def _wait_loss(self, micro_iter_idx):
        fut = self._loss_fut.get(micro_iter_idx)
        if fut is None:
            fut = torch.futures.Future()
            self._loss_fut[micro_iter_idx] = fut
        loss = fut.wait()
        self._loss_fut.pop(micro_iter_idx, None)
        return loss
        
    def base_forward_only(self, epoch_idx, iter_idx, micro_iter_idx, inputs=None, pull_from=None):            
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
            # 优先使用直传 inputs（首段/调试）；否则按令牌去上游取
            with self._mb_lock:
                if micro_iter_idx in self._fwd_seen:
                    return
                self._fwd_seen.add(micro_iter_idx)
            
            if inputs is not None:
                if not isinstance(self.base_inputs, dict):
                    self.base_inputs = {}
                # print("self.device: ", self.device)
                self.base_inputs[micro_iter_idx] = to_device(inputs, self.device)
            elif pull_from is not None:
                prev_worker_name, mb = pull_from
                # 直接向上游 worker 拉取该 micro-batch 的前向输出（单跳，驱动不搬数据）
                fetched = global_shared_rrefs[prev_worker_name].rpc_sync().fetch_forward_out(mb)
                if self.network.include_embed:
                    self.base_inputs = to_device(fetched, self.device)
                else:
                    if not isinstance(self.base_inputs, dict): self.base_inputs = {}
                    self.base_inputs[micro_iter_idx] = to_device(fetched, self.device)
            
            if self.network.include_lm_head:  #最后一个stage
                self.base_inputs[micro_iter_idx]['hidden_states'] = self.base_inputs[micro_iter_idx]['hidden_states'].detach().requires_grad_()
                logits = self.network(**self.base_inputs[micro_iter_idx], use_cache=False)
                loss = self.criterion(logits=logits, labels=self.labels[micro_iter_idx], 
                                                                vocab_size=self.args.vocab_size) / self.args.micro_batch_num
                self.base_loss[micro_iter_idx] = loss
                self._stash_loss(micro_iter_idx, loss)
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
                # self.base_inputs = to_device(inputs, self.device)
                self.base_outputs[micro_iter_idx] = self.network(**self.base_inputs[micro_iter_idx], use_cache=False)
                self._stash_forward_out(micro_iter_idx, self.base_outputs[micro_iter_idx])
                # global_shared_rrefs[self.next_device_name].rpc_sync().base_forward_receive(micro_iter_idx, to_cpu(self.base_outputs[micro_iter_idx]))
            else:
                self.base_inputs[micro_iter_idx]['hidden_states'] = self.base_inputs[micro_iter_idx]['hidden_states'].detach().requires_grad_()
                self.base_outputs[micro_iter_idx] = self.network(**self.base_inputs[micro_iter_idx], use_cache=False)
                self._stash_forward_out(micro_iter_idx, self.base_outputs[micro_iter_idx])
                # global_shared_rrefs[self.next_device_name].rpc_sync().base_forward_receive(micro_iter_idx, to_cpu(self.base_outputs[micro_iter_idx]))
    
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
        if micro_iter_idx in self._bwd_seen:
            return
        self._bwd_seen.add(micro_iter_idx)
        
        if self.network.include_lm_head:
            loss = self.base_loss.get(micro_iter_idx)
            if loss is None:
                loss = self._wait_loss(micro_iter_idx)     # <<< 等待前向/算 loss
                self.base_loss[micro_iter_idx] = loss

            self.base_loss[micro_iter_idx].backward()
            del self.base_loss[micro_iter_idx]
            if ((not self.iszeroth) and (not self.iszeroth_ls[self.previous_device_idx])) and self.device_idx != 0:
                grad_to_prev = self.base_inputs[micro_iter_idx]['hidden_states'].grad.detach().cpu()
                sig = global_shared_rrefs[self.previous_device_name].rpc_async() \
                    .base_backward_receive(micro_iter_idx, grad_to_prev)
                del self.base_inputs[micro_iter_idx]
                sig.wait()
        else:
            # 等待来自下游的梯度（若反向先触发，会在此阻塞而不是报错）
            fut = self._bwd_inbox_fut.get(micro_iter_idx)
            if fut is None:
                fut = torch.futures.Future()
                self._bwd_inbox_fut[micro_iter_idx] = fut
            grad_out = fut.wait()
            self._bwd_inbox_fut.pop(micro_iter_idx, None)

            self.base_outputs[micro_iter_idx]['hidden_states'].backward(grad_out)
            del self.base_outputs[micro_iter_idx]

            # 继续把梯度发回上一段（若需要）
            if ((not self.iszeroth) and (not self.iszeroth_ls[self.previous_device_idx])) and self.device_idx != 0:
                grad_to_prev = self.base_inputs[micro_iter_idx]['hidden_states'].grad.detach().cpu()
                sig = global_shared_rrefs[self.previous_device_name].rpc_async() \
                    .base_backward_receive(micro_iter_idx, grad_to_prev)
                del self.base_inputs[micro_iter_idx]
                sig.wait()

        torch.cuda.empty_cache()
        
            
        # else:
        #     if micro_iter_idx not in self.output_gradients:
        #         raise RuntimeError(f"missing output_gradients for mb={micro_iter_idx} on {rpc.get_worker_info().name}")
        #     grad_out = self.output_gradients.pop(micro_iter_idx)

        #     self.base_outputs[micro_iter_idx]['hidden_states'].backward(grad_out)
        #     del self.base_outputs[micro_iter_idx]
        
        #     # self.base_outputs[micro_iter_idx]['hidden_states'].backward(self.output_gradients[micro_iter_idx])
        #     # del self.base_outputs[micro_iter_idx]
        
        # if ((not self.iszeroth) and (not self.iszeroth_ls[self.previous_device_idx])) and self.device_idx != 0:
        #     tmp_send_signals = \
        #         global_shared_rrefs[self.previous_device_name].rpc_async().base_backward_receive(micro_iter_idx, 
        #                                                                                         self.base_inputs[micro_iter_idx]['hidden_states'].grad.cpu())
        #     del self.base_inputs[micro_iter_idx]
        #     tmp_send_signals.wait()
        # torch.cuda.empty_cache()
    
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
                elif self.last_zeroth_model:
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
                    print("self.estimated_loss: ",self.estimated_loss)
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
                        # self.loss_writer.add_histogram("estimate_loss", torch.tensor(self.estimated_loss_ls), global_step=iter_idx)
                else:
                    # self.estimated_inputs = to_device(inputs, self.device)
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
                print("self.estimated_loss: {}, base_loss_eval: {}".format(self.estimated_loss, base_loss_eval))
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
