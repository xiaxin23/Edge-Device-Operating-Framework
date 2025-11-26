import io
import math
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification    # 注意：你的环境里是 CausalLM
from transformers.optimization import get_scheduler
from models.pipeline_modeling_llama import llama_stage, seqcls_llama_stage
from loss_utils import ForCausalLMLoss, ForSequenceClassificationLoss
from accelerate import init_empty_weights
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
import torch, re
from safetensors.torch import safe_open

import re
from typing import Optional, Sequence, Union
import torch
from safetensors.torch import safe_open, save_file
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from accelerate import init_empty_weights 
import torch.distributed as dist

try:
    import torch_npu
    HAS_TORCH_NPU = True
except ModuleNotFoundError:
    HAS_TORCH_NPU = False

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
                    # subset_sd["score.weight"] = _maybe_cast_to_dtype(f.get_tensor(k))
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

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.lora_A = nn.Parameter(torch.empty(in_dim, rank, dtype=torch.bfloat16))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim, dtype=torch.bfloat16))
        self.alpha = alpha
    def forward(self, x):
        return self.alpha * (x @ self.lora_A @ self.lora_B)

class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)
    def forward(self, x):
        return self.linear(x) + self.lora(x)

def replace_linear_with_lora(model, r, alpha):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and ("q_proj" in name or "v_proj" in name):
            setattr(model, name, LinearWithLoRA(module, r, alpha))
        else:
            replace_linear_with_lora(module, r, alpha)

def to_device(obj, device="cuda"):
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [to_device(v, device) for v in obj]
        return type(obj)(t)
    try:
        return obj.to(device)
    except Exception:
        return obj

def to_cpu(obj):
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        t = [to_cpu(v) for v in obj]
        return type(obj)(t)
    try:
        return obj.cpu()
    except Exception:
        return obj

def make_deterministic_weight(shape, seed=0, std=0.02, dtype=torch.bfloat16, device="cpu"):
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    w = torch.randn(shape, generator=g, device="cpu", dtype=torch.float32) * std
    return w.to(dtype=dtype, device=device)

class DistributedSubModel:
    def __init__(self, args, model_split_config, iszeroth_ls, device, stage_idx: int, iszeroth: bool, total_stages: int):
        self.args = args
        self.model_split_config = model_split_config
        self.stage_idx = stage_idx
        self.iszeroth = iszeroth
        self.total_stages = total_stages
        self.iszeroth_ls = iszeroth_ls

        self.device = device
        self.include_embed = (stage_idx == 0)
        self.include_lm_head = (stage_idx == total_stages - 1)
        self.last_zeroth_model = True if self.iszeroth and (not self.iszeroth_ls[(self.stage_idx+1)%len(self.iszeroth_ls)]) else False
        # 你此前的假设：zeroth 是从 stage0 开始的连续前缀
        # 因此最后一个 zeroth 段 = 前缀末尾；运行时由主控决定谁是“last_zeroth”，本类不需要保存索引。

        # self.generate_subnetwork()
        self._build_model()
        if self.args.dataset == "agnews":
            self.criterion = ForSequenceClassificationLoss
        elif self.args.dataset == "math":
            self.criterion = ForCausalLMLoss

        self.optimizer = torch.optim.Adam(
            (p for p in self.network.parameters() if p.requires_grad),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )
        self.lr_scheduler = get_scheduler(
            "cosine",
            optimizer=self.optimizer,
            num_warmup_steps=int(0.1*self.args.epoch*self.args.total_iterations),
            num_training_steps=int(self.args.epoch*self.args.total_iterations)
        )

        # 训练时态
        self.labels = {}               # mb -> labels（仅 last stage / last zeroth 需要）
        self.base_inputs = {}          # mb -> dict
        self.base_outputs = {}         # mb -> dict with 'hidden_states'
        self.base_loss = {}            # mb -> scalar loss (last stage)
        self.base_loss_eval = {}       # mb -> float
        self.intermidate_base_loss = {}

        # 估计用
        self.params_dict = None
        self.estimated_inputs = {}     # mb -> dict[q] -> inputs
        self.estimated_loss = {}       # mb -> dict[q] -> float
        self.directional_derivative = {}  # mb -> dict[q] -> float
        self.perturbs_dict = {}        # mb -> dict[q] -> {name: perturb}
        
        # self.prepre_fietune()

    def pre_finetune(self, s, base_inputs=None):
        base_forward_total_steps = 3 + 4 - 1
        base_backward_total_steps = 3 + 4 - 1
        total_steps = 2 * (3 + 4 - 1)
        for step_idx in range(base_forward_total_steps):
            # print(f"step {step_idx} start!!")
            if step_idx < base_forward_total_steps:
                forward_mb_num = step_idx - s
                if 0 <= forward_mb_num < 4:
                    # print("stage: {}, forward mb: {} done!!".format(s, forward_mb_num))
                    if s == 0:
                        out = self.base_forward_only(0, 1, forward_mb_num, inputs=base_inputs)
                        # print("out: ", out)
                        # mem_bytes = get_mem(out)
                        # print(f"forward {mem_bytes / 1024**2:.4f} MB")
                        if s < 3:
                            send_obj(out, dst=s+1, tag=tag_fwd(forward_mb_num))
                    else:
                        inp = recv_obj(src=s-1, tag=tag_fwd(forward_mb_num))
                        # print("inp: ", inp)
                        # import sys
                        # sys.exit(0)
                        out = self.base_forward_only(0, 1, forward_mb_num, inputs=inp)
                        if s < 3:
                            send_obj(out, dst=s+1, tag=tag_fwd(forward_mb_num))

    def prepre_fietune(self):
        if self.iszeroth:
            tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            self.set_train_configuration()
            print("pretraining!!!")
            if self.include_embed:
                prompt = "你好"
                inputs = tokenizer(prompt, return_tensors="pt")
                inputs = {'input_ids': inputs['input_ids'].npu(), 'attention_mask': inputs['attention_mask'].npu()}
                print(inputs)
            self.network.eval()
            if self.include_embed:
                self.pre_finetune(0, inputs)
            elif self.last_zeroth_model:
                self.pre_finetune(2)
            elif self.include_embed:
                self.pre_finetune(1)
        dist.barrier()
        self.network.train()
        
    def _build_model(self):
        if self.args.dataset == "agnews":
            self.config = AutoConfig.from_pretrained(self.args.model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
            label2id = {v: k for k, v in id2label.items()}
            self.config.num_labels = len(label2id)
            self.config.id2label = id2label
            self.config.label2id = label2id
            self.config.attn_implementation = "eager"
            self.config.pad_token_id = tokenizer.pad_token_id

            # 1) 用“空权重”上下文构空壳模型（不会传 empty_init）
            with init_empty_weights():
                full = AutoModelForSequenceClassification.from_config(
                    self.config, torch_dtype=torch.bfloat16
                )

            # 2) 构建你的 stage 子模型（空壳）
            self.network = seqcls_llama_stage(
                base_model=full,
                layer_start=self.model_split_config[0],
                layer_end=self.model_split_config[1],
                include_embed=self.include_embed,
                include_lm_head=self.include_lm_head,
                is_zeroth=self.iszeroth,
                last_zeroth_model=self.last_zeroth_model,
            )
            self.network = self.network.to_empty(device="cpu")  # 仍是空权重
            self.network.rotary_emb = LlamaRotaryEmbedding(self.config)

            # 3) 只加载需要的 LLaMA 底座权重
            sd = extract_llama_subset_safetensors(
                ckpt_path=f"{self.args.model_name_or_path}/model.safetensors",
                keep_layers=(self.model_split_config[0], self.model_split_config[1]),
                include_embed=self.include_embed,
                include_norm=True if self.include_lm_head or self.last_zeroth_model else False,
                include_lm_head=False,                 # 注意：分类头不是 lm_head
                dtype=torch.bfloat16,
                out_path=None,
            )

            # 4) 分类头是 `score`（Linear(hidden_size -> num_labels)）
            #    权重形状应为 (num_labels, hidden_size)，不要写反
            if self.include_lm_head or self.last_zeroth_model:
                num_labels  = self.config.num_labels
                hidden_size = self.config.hidden_size
                sd["score.weight"] = make_deterministic_weight(
                    (num_labels, hidden_size),
                    seed=getattr(self.args, "seed", 0),
                    std=0.02, dtype=torch.bfloat16, device="cpu"
                )
                # 如果实现包含 bias（LLaMA 通常没有），也一并初始化：
                # sd["score.bias"] = torch.zeros(num_labels, dtype=torch.bfloat16)

            unexp, miss = self.network.load_state_dict(sd, strict=False)
            print(f"unexpect: {unexp}, missing: {miss}")

            self.network.config.pad_token_id = tokenizer.pad_token_id

            # 5) LoRA 只训 LoRA 参数
            replace_linear_with_lora(self.network, self.args.lora_r, self.args.lora_alpha)
            for n, p in self.network.named_parameters():
                if "lora" not in n:
                    p.requires_grad = False

            self.network = self.network.to(self.device)            
        elif self.args.dataset == "multirc":
            pass
        elif self.args.dataset == "squad":
            pass
        elif self.args.dataset == "math":
            self.args.vocab_size = 128258
            self.config = AutoConfig.from_pretrained(self.args.model_name_or_path)
            with torch.device("meta"):
                full = AutoModelForCausalLM.from_pretrained(
                    self.args.model_name_or_path,
                    config=self.config,
                    ignore_mismatched_sizes=True,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    attn_implementation="eager"
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
            # del full
            # torch.cuda.empty_cache()
            self.network = self.network.to_empty(device="cpu")
            self.network.rotary_emb = LlamaRotaryEmbedding(self.config)
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
        self.network.train()
        
    def generate_subnetwork(self):
        print("KKKKK generate_subnetwork")
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
        # torch.cuda.empty_cache()
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

    def set_train_configuration(self):
        self.optimizer.zero_grad(set_to_none=True)
        self.network.train()

        self.base_inputs.clear()
        self.base_outputs.clear()
        self.base_loss.clear()
        self.base_loss_eval.clear()
        self.intermidate_base_loss.clear()

        self.estimated_inputs = {mb: {} for mb in range(self.args.micro_batch_num)}
        self.estimated_loss   = {mb: {} for mb in range(self.args.micro_batch_num)}
        self.directional_derivative = {mb: {} for mb in range(self.args.micro_batch_num)}
        self.perturbs_dict    = {mb: {} for mb in range(self.args.micro_batch_num)}

        if self.iszeroth:
            self.params_dict = {name: p for name, p in self.network.named_parameters() if p.requires_grad}
        else:
            self.params_dict = None

    # ----------- base forward/backward -----------

    def base_forward_only(self, epoch_idx, iter_idx, mb, inputs):
        if self.iszeroth:
            with torch.no_grad():
                inputs = to_device(inputs, self.device)
                outs = self._f_zeroth(self.params_dict, inputs, mb, labels=self.labels.get(mb, None))
                if self.include_lm_head:   # 极端情况（全是 zeroth；一般不会）
                    self.base_loss_eval[mb] = float(outs.detach().cpu().item())
                    return None
                else:
                    return to_cpu(outs)
        else:
            if self.include_embed:
                x = to_device(inputs, self.device)
                y = self.network(**x, use_cache=False)
                self.base_outputs[mb] = y
                return to_cpu(y)
            elif self.include_lm_head:
                # 末段：需要可求导
                self.base_inputs[mb] = to_device(inputs, self.device)
                self.base_inputs[mb]['hidden_states'] = self.base_inputs[mb]['hidden_states'].detach().requires_grad_()
                if self.args.dataset == "agnews":
                    logits, pooled_logits = self.network(**self.base_inputs[mb], use_cache=False)
                    self.base_loss[mb] = self.criterion(labels=self.labels[mb], pooled_logits=pooled_logits, config=self.config) / self.args.micro_batch_num 
                elif self.args.dataset == "math":
                    logits = self.network(**self.base_inputs[mb], use_cache=False)
                    self.base_loss[mb] = self.criterion(logits=logits, labels=self.labels[mb], vocab_size=self.args.vocab_size) / self.args.micro_batch_num
                del logits
                # torch.cuda.empty_cache()
                self.base_loss_eval[mb] = self.base_loss[mb].detach().cpu().item()
                if mb == self.args.micro_batch_num - 1:
                    mean_loss_value = sum(list(self.base_loss_eval.values()))
                    print("mean_loss_value: ", mean_loss_value)
                return None
            else:
                self.base_inputs[mb] = to_device(inputs, self.device)
                self.base_inputs[mb]['hidden_states'] = self.base_inputs[mb]['hidden_states'].detach().requires_grad_()
                y = self.network(**self.base_inputs[mb], use_cache=False)
                self.base_outputs[mb] = y
                return to_cpu(y)

    def base_backward_only(self, mb, grad_in=None):
        if self.include_lm_head:
            self.base_loss[mb].backward()
            del self.base_loss[mb]
            # torch.cuda.empty_cache()
            return to_cpu(self.base_inputs[mb]['hidden_states'].grad)
        else:
            grad_in = to_device(grad_in, self.device)
            self.base_outputs[mb]['hidden_states'].backward(grad_in)
            del self.base_outputs[mb]
            if self.stage_idx > 0:
                gprev = self.base_inputs[mb]['hidden_states'].grad
                del self.base_inputs[mb]
                # torch.cuda.empty_cache()
                return to_cpu(gprev)
            # torch.cuda.empty_cache()
            return None

    # ----------- estimate 链（zeroth 前缀 -> 非 zeroth -> 末段算估计损失） -----------

    @torch.no_grad()
    def estimate_grads(self, epoch_idx, iter_idx, mb, q, inputs):
        if self.iszeroth:
            if self.include_embed:
                if q == 0:
                    self.estimated_inputs[mb] = {}
                    base_in = to_device(inputs, self.device)
                    self.estimated_inputs[mb]['_stage0_cache'] = base_in
                base_in = self.estimated_inputs[mb].get('_stage0_cache')
                perturbs, perturbed = self._sample_perturb_params()
                self.perturbs_dict[mb][q] = perturbs
                est_out = self._f_raw(perturbed, base_in)  # 只 forward 到输出
                return to_cpu(est_out)
            elif self.last_zeroth_model and q >= 0:
                perturbs, perturbed = self._sample_perturb_params()
                self.perturbs_dict[mb][q] = perturbs
                x = to_device(inputs, self.device)
                # output, logits, pooled_logits = self._f_raw(perturbed, x)
                if self.args.dataset == "agnews":
                    output, logits, pooled_logits = self._f_raw(perturbed, x)
                    est_loss = self.criterion(labels=self.labels[mb], pooled_logits=pooled_logits, config=self.config)
                elif self.args.dataset == "math":
                    output, logits = self._f_raw(perturbed, x)
                    est_loss = self.criterion(logits=logits, labels=self.labels[mb], vocab_size=self.args.vocab_size)
                self.estimated_loss[mb][q] = float(est_loss.detach().cpu().item())
                return None
            else:
                # 中间 zeroth 段
                perturbs, perturbed = self._sample_perturb_params()
                self.perturbs_dict[mb][q] = perturbs
                x = to_device(inputs, self.device)
                y = self._f_raw(perturbed, x)
                return to_cpu(y)
        else:
            # 非 zeroth：不扰动参数，沿链传递
            if self.include_lm_head:
                x = to_device(inputs, self.device)
                if self.args.dataset == "agnews":
                    logits, pooled_logits = self.network(**x, use_cache=False)
                    est_loss = self.criterion(labels=self.labels[mb], pooled_logits=pooled_logits, config=self.config)
                elif self.args.dataset == "math":
                    logits = self.network(**x, use_cache=False)
                    est_loss = self.criterion(logits=logits, labels=self.labels[mb], vocab_size=self.args.vocab_size)
                self.estimated_loss[mb][q] = float(est_loss.detach().cpu().item())
                return None
            else:
                x = to_device(inputs, self.device)
                y = self.network(**x, use_cache=False)
                return to_cpu(y)

    # 末端把 base_loss_eval 发回后，由“最后一个 zeroth 段”调用
    def zeroth_compute_directional_derivative(self, base_loss_eval: dict):
        # 每个 (mb,q) -> (est - base) / step_size
        for mb in range(self.args.micro_batch_num):
            for q in range(self.args.q_num):
                dd = (self.estimated_loss[mb][q] - base_loss_eval[mb]) / self.args.zo_step_size
                self.directional_derivative[mb][q] = float(dd)
        return self.directional_derivative

    # 所有 zeroth 段应用导数，聚合出 param.grad
    def update_zeroth_model(self, directional_derivatives: dict):
        grads = {}
        for mb in range(self.args.micro_batch_num):
            for q in range(self.args.q_num):
                dd = directional_derivatives[mb][q]
                for name, perturb in self.perturbs_dict[mb][q].items():
                    g = perturb * dd / self.args.q_num
                    if name in grads:
                        grads[name] += g
                    else:
                        grads[name] = g
        for name, p in self.params_dict.items():
            p.grad = (grads[name] / self.args.micro_batch_num).to(p.dtype).to(p.device)

    def network_optimizer(self):
        self.optimizer.step()
        self.lr_scheduler.step()

    # ----------- 内部小工具 -----------

    @torch.no_grad()
    def _f_raw(self, params_dict, inputs):
        backup = {n: t.clone() for n, t in self.network.named_parameters() if t.requires_grad}
        self.network.load_state_dict(params_dict, strict=False)
        out = self.network(**inputs, use_cache=False)
        self.network.load_state_dict(backup, strict=False)
        return out

    @torch.no_grad()
    def _f_zeroth(self, params_dict, inputs, mb, labels=None):
        backup = {n: t.clone() for n, t in self.network.named_parameters() if t.requires_grad}
        self.network.load_state_dict(params_dict, strict=False)
        if self.last_zeroth_model:
            if self.args.dataset == "agnews":
                out, logits, pooled_logits = self.network(**inputs, use_cache=False)
            elif self.args.dataset == "math":  
                out, logits = self.network(**inputs, use_cache=False)
        else:
            out = self.network(**inputs, use_cache=False)
        self.network.load_state_dict(backup, strict=False)
        if labels is not None:
            if self.args.dataset == "agnews":
                loss = self.criterion(labels=labels, pooled_logits=pooled_logits, config=self.config)
            elif self.args.dataset == "math":   
                loss = self.criterion(logits=logits, labels=labels, vocab_size=self.args.vocab_size)
            del logits
            # torch.cuda.empty_cache()
            self.intermidate_base_loss[mb] = loss
        return out

    def _sample_perturb_params(self):
        perturbs, perturbed = {}, {}
        for name, p in self.params_dict.items():
            e = torch.randn_like(p)
            e = e / (torch.norm(e) + 1e-8)
            perturbs[name] = e
            perturbed[name] = p + self.args.zo_step_size * e
        return perturbs, perturbed
