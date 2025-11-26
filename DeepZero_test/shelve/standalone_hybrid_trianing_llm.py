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
    
def parse_args():
    parser = argparse.ArgumentParser("Pretraining")
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--model', choices=['vit', 'llama', 'qwen'], default='llama')
    parser.add_argument('--model_name_or_path', type=str, default="/home/liyan/Llama-3.2-3B/")
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--device', type=int, default=2)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1234567)
    parser.add_argument('--method', choices=['rge', 'bge', 'cge', 'no', 'hybrid'], default='no')
    parser.add_argument('--dataset', choices=['cifar10','mnist','math'], default='cifar10')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--zo_step_size', type=float, default=1e-3)
    parser.add_argument('--q_num', type=int, default=3)
    parser.add_argument('--epoch', type=int, default=4)
    parser.add_argument('--warm-epochs', type=int, default=3)
    
    args = parser.parse_args()
    return args
    
@torch.no_grad()
def f(params_dict, network, inputs, labels=None, loss_func=None):
    state_dict_backup = {
        name: param.clone() for name, param in network.named_parameters() if param.requires_grad
    }
    network.load_state_dict(params_dict, strict=False)
    output = network(**inputs, use_cache=False)
    network.load_state_dict(state_dict_backup,strict=False)
    del state_dict_backup
    torch.cuda.empty_cache()
    if labels is not None and loss_func is not None:
        logits = output.logits
        loss = loss_func(logits=logits, labels=labels, vocab_size=vocab_size)
        del output, logits
        torch.cuda.empty_cache()
        return loss
    else:
        return output

def generate_subnetworks(args, model_split_configs, iszeroth_ls, total_devices):
    sub_network_ls = []
    with torch.device("meta"):
        network = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=model_config,
            ignore_mismatched_sizes=True,
            # cache_dir="./cache",
            torch_dtype=torch.bfloat16
        )
        for stage_idx, split_idx in enumerate(model_split_configs):
            subnetwork = llama_stage(base_model=network, 
                                        layer_start=model_split_configs[stage_idx][0], 
                                        layer_end=model_split_configs[stage_idx][1],
                                        include_embed=True if stage_idx == 0 else False,
                                        include_lm_head=True if stage_idx == total_devices-1 else False,
                                        is_zeroth=iszeroth_ls[stage_idx])
            subnetwork = subnetwork.to_empty(device="cpu")
            # for name, param in subnetwork.named_parameters():
            #     print(name)
            # print("*"*50)
            if stage_idx == 0:
                embe_token = torch.load("./model_part_layers/llama3b/embedding.pt")#.to(device)
                embe_token = {".".join(k.split(".")[1:]): v for k, v in embe_token.items()}
                subnetwork.load_state_dict(embe_token, strict=False, assign=True)
            
            for layer_idx in range(model_split_configs[stage_idx][0], model_split_configs[stage_idx][1]):
                layer_tensors = torch.load("./model_part_layers/llama3b/layer_"+str(layer_idx)+".pt")#.to(device)
                layer_tensors  = {"layers."+str(layer_idx-model_split_configs[stage_idx][0])+"."+k: v for k, v in layer_tensors.items()}
                subnetwork.load_state_dict(layer_tensors, strict=False, assign=True)
            
            if stage_idx == total_devices-1:
                head_norm = torch.load("./model_part_layers/llama3b/head_norm.pt")#.to(device)
                head_norm = {"norm.weight": v for k, v in head_norm.items()}
                subnetwork.load_state_dict(head_norm, strict=False, assign=True)
                embe_token = torch.load("./model_part_layers/llama3b/embedding.pt")#.to(device)
                embe_token = {"lm_head.weight": v for k, v in embe_token.items()}
                subnetwork.load_state_dict(embe_token, strict=False, assign=True)
            
            sub_network_ls.append(subnetwork)
    return sub_network_ls

def compare_tensor_dicts(dict1, dict2):
    if dict1.keys() != dict2.keys():
        return False
    for key in dict1:
        if not torch.equal(dict1[key], dict2[key]):
            return False
    return True

def compare_estimated_outputs(estimated_outputs_ls):
    for i in range(1, len(estimated_outputs_ls)):
        if not compare_tensor_dicts(estimated_outputs_ls[0], estimated_outputs_ls[i]):
            return False
    return True

    
'''
def llm_finetuning(args, model, train_loaders, device):
    criterion = ForCausalLMLoss_chunked
    # test(model, test_data, device)
    print(args.method)
    for epoch in range(args.epochs):
        model.train()
        print('Epoch: ', epoch)
        if args.method != 'no':
            # print("kkk")
            # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=5e-4)
            # global_length = (args.epoch - args.warm_epochs) * len(train_data)
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=global_length)
            with torch.no_grad():
                for idx, (inputs,labels) in tqdm(enumerate(train_loaders)):
                    optimizer.zero_grad()
                    inputs = to_device(inputs, device)
                    labels = to_device(labels, device)
                    f_theta = partial(f, network=model, inputs=inputs, labels=labels, loss_func=criterion)
                    params_dict = {
                        name: p for name, p in model.named_parameters() if p.requires_grad
                        }
                    if args.method == "rge":
                        grads_dict = rge(f_theta, params_dict, args.q_num, args.zo_step_size)
                    elif args.method == "cge":
                        grads_dict = cge(f_theta, params_dict, args.zo_step_size)
                    elif args.method == "bge":
                        grads_dict = bge(f_theta, params_dict, args.q_num, args.zo_step_size)
                    for key, param in params_dict.items():
                        param.grad = grads_dict[key]
                    if idx % 500 == 0:
                        # print("jjjj")
                        for name,param in model.named_parameters():
                            if param.requires_grad:
                                avg_abs, mode_magnitude = tensor_abs_stats(param.grad)
                                print("name: {}, avg_abs: {}, mode_magnitude: {}".format(name, avg_abs, mode_magnitude))
                    optimizer.step()
                    # if epoch >= args.warm_epochs:
                    #     scheduler.step()
                    # break
            # model = zero_train(args, model, train_data, device, epoch)
        else:
            # print("jjjj")
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
            criterion = ForCausalLMLoss_chunked
            # 开始训练 
            model.train()
            for idx, (inputs,labels) in tqdm(enumerate(train_loaders)):
                optimizer.zero_grad()
                inputs = to_device(inputs, device)
                labels = to_device(labels, device)
                out = model(**inputs, use_cache=False)
                # predict = torch.argmax(out, dim=-1)
                loss = out.loss
                loss.backward()
                if idx % 500 == 0:
                    print("jjjj")
                    for name,param in model.named_parameters():
                        if param.requires_grad:
                            # print(name, torch.mean(torch.abs(param.grad)))
                            avg_abs, mode_magnitude = tensor_abs_stats(param.grad)
                            print("name: {}, avg_abs: {}, mode_magnitude: {}".format(name, avg_abs, mode_magnitude))
                optimizer.step() 
            # test(model, test_data, device)
            # torch.save(model.state_dict(), f"./.cache/vit_model.pth")
            # break
        # test(model, test_data, device)
'''

def llm_finetuning(args, model_ls, train_loaders, device):   
    optimizer = torch.optim.Adam(
        (p for model in model_ls for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=5e-4
    )
    # global_length = (args.epoch - args.warm_epochs) * len(train_data)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=global_length)
    for epoch in range(args.epochs):
        print('Epoch: {}, len(train_loaders): {}'.format(epoch, len(train_loaders)))
        iter_time = 0 
        for idx, (inputs, labels) in enumerate(train_loaders):
            if idx < 50:
                start = time.time()
            optimizer.zero_grad()
            perturbs_dict_models_ls = []
            directional_derivative_ls = []
            base_outputs = None
            estimated_outputs_ls = []
            for model_idx in range(len(model_ls)):
                model_ls[model_idx].train()
                if model_ls[model_idx].is_zeroth:    #仅前向传播
                    perturbs_dict_ls = []
                    with torch.no_grad():
                        if model_ls[model_idx].include_embed:  #第一个stage
                            inputs = to_device(inputs, device)
                            params_dict = {
                                name: p for name, p in model_ls[model_idx].named_parameters() if p.requires_grad
                                }
                            f_theta = partial(f, network=model_ls[model_idx], inputs=inputs)
                            base_outputs = f_theta(params_dict)
                            if args.method == "rge":
                                for _ in range(args.q_num):
                                    perturbs_dict, perturbed_params_dict = {}, {}
                                    for key, param in params_dict.items():
                                        # perturb = torch.randn_like(param)
                                        # perturb /= (torch.norm(perturb) + 1e-8)
                                        perturb = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                                        perturbs_dict[key] = perturb
                                        # print("perturbs_dict[{}]: {}".format(key, perturbs_dict[key]))
                                        perturbed_params_dict[key] = args.zo_step_size * perturb + param
                                    perturbs_dict_ls.append(perturbs_dict)
                                    estimated_outputs = f_theta(perturbed_params_dict)
                                    estimated_outputs_ls.append(estimated_outputs)
                            elif args.method == "cge":
                                pass
                            elif args.method == "bge":
                                pass
                        else:
                            base_inputs = to_device(base_outputs, device)
                            del base_outputs
                            torch.cuda.empty_cache()
                            params_dict = {
                                name: p for name, p in model_ls[model_idx].named_parameters() if p.requires_grad
                                }
                            f_theta = partial(f, network=model_ls[model_idx])
                            base_outputs = f_theta(params_dict, base_inputs)
                            for estimated_outputs_idx in range(len(estimated_outputs_ls)):
                                estimated_outputs_ls[estimated_outputs_idx] = to_device(estimated_outputs_ls[estimated_outputs_idx], device)
                                # f_theta = partial(f, network=model_ls[model_idx], inputs=estimated_outputs_ls[estimated_outputs_idx])
                                if args.method == "rge":
                                    perturbs_dict, perturbed_params_dict = {}, {}
                                    for key, param in params_dict.items():
                                        # perturb = torch.randn_like(param)
                                        # perturb /= (torch.norm(perturb) + 1e-8)
                                        perturb = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
                                        perturbs_dict[key] = perturb
                                        perturbed_params_dict[key] = args.zo_step_size * perturb + param
                                    perturbs_dict_ls.append(perturbs_dict)
                                    estimated_outputs_ls[estimated_outputs_idx] = f_theta(perturbed_params_dict, estimated_outputs_ls[estimated_outputs_idx])
                                elif args.method == "cge":
                                    pass
                                elif args.method == "bge":
                                    pass
                        # print(compare_estimated_outputs(estimated_outputs_ls))
                        # import sys
                        # sys.exit(0)
                    perturbs_dict_models_ls.append(perturbs_dict_ls)
                else:   #forward and backward pass
                    if model_ls[model_idx].include_lm_head:  #最后一个stage
                        base_inputs = to_device(base_outputs, device)
                        del base_outputs
                        torch.cuda.empty_cache()
                        labels = to_device(labels, device)
                        criterion = ForCausalLMLoss
                        
                        logits = model_ls[model_idx](**base_inputs, use_cache=False)
                        base_loss = criterion(logits=logits, labels=labels, vocab_size=vocab_size)
                        base_loss_eval = base_loss.detach().item()
                        print("base_loss: ", base_loss_eval)
                        base_loss.backward()
                        del base_inputs, logits
                        torch.cuda.empty_cache()
                        with torch.no_grad(): 
                            for estimated_outputs in estimated_outputs_ls:
                                estimated_outputs = to_device(estimated_outputs, device)
                                estimated_logits = model_ls[model_idx](**estimated_outputs, use_cache=False)
                                # logits = estimated_outputs.logits
                                esitmated_loss = criterion(logits=estimated_logits, labels=labels, vocab_size=vocab_size).detach().item()
                                print("esitmated_loss: ", esitmated_loss)
                                directional_derivative = (esitmated_loss - base_loss_eval) / args.zo_step_size
                                print("directional_derivative: ", directional_derivative)
                                directional_derivative_ls.append(directional_derivative)       
                    else:
                        base_inputs = to_device(base_outputs, device)
                        del base_outputs
                        torch.cuda.empty_cache()
                        base_outputs = model_ls[model_idx](**base_inputs, use_cache=False)
                        with torch.no_grad(): 
                            for estimated_outputs_idx in range(len(estimated_outputs_ls)):
                                estimated_outputs_ls[estimated_outputs_idx] = to_device(estimated_outputs_ls[estimated_outputs_idx], device)
                                estimated_outputs_ls[estimated_outputs_idx] = model_ls[model_idx](**estimated_outputs_ls[estimated_outputs_idx], use_cache=False)
            
            #zeroth-order update
            for model_idx in range(len(model_ls)):
                if not model_ls[model_idx].is_zeroth:
                    break
                params_dict = {
                    name: p for name, p in model_ls[model_idx].named_parameters() if p.requires_grad
                    }
                grads_dict = {}
                perturbs_dict_ls = perturbs_dict_models_ls[model_idx]
                for perturbs_dict, directional_derivative in zip(perturbs_dict_ls, directional_derivative_ls):
                    if len(grads_dict.keys()) == len(params_dict.keys()):
                        for key, perturb in perturbs_dict.items():
                            grads_dict[key] += perturb * directional_derivative / args.q_num
                    else:
                        for key, perturb in perturbs_dict.items():
                            grads_dict[key] = perturb * directional_derivative / args.q_num
                for key, param in params_dict.items():
                    param.grad = grads_dict[key]
                    # print(param.grad)
                        
            if idx % 100 == 0:
                for model_idx in range(len(model_ls)):
                    for name,param in model_ls[model_idx].named_parameters():
                        if param.requires_grad:
                            # print(name, torch.mean(torch.abs(param.grad)))
                            avg_abs, mode_magnitude = tensor_abs_stats(param.grad)
                            print("name: {}, avg_abs: {}, mode_magnitude: {}".format(name, avg_abs, mode_magnitude))              
                            
            optimizer.step()
            if idx < 50:
                end = time.time()
                iter_time += (end-start)
                if idx == 49:
                    print("time(s)/it: ", iter_time/50)


if __name__ == "__main__":
    args = parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    ours_set_seed(args.seed)    
    set_seed(args.seed)
    device = f"cuda:{args.device}"
    
    if args.model == "vit":
        network_init_func = vit_lora   #vit_with_classifiers
        network_kwargs = {'num_classes': 10}
        network = network_init_func(**network_kwargs)
        # time.sleep(100)
        param_dict = torch.load(f"./model_store/vit_model.pth", map_location=torch.device('cpu'))
        network.load_state_dict(param_dict, strict=False)
        del param_dict
        loaders, class_num = prepare_dataset(args.dataset, args.batch_size, device=device)
        for key,param in network.named_parameters():
            if '_a_' in key or '_b_' in key:
                continue
            else:
                param.requires_grad = False
        network = network.to(device)
    else:   
        #load tokenizer     
        tokenizer = load_hf_tokenizer(args.model_name_or_path,
                                  fast_tokenizer=True,
                                  model_max_length=args.max_seq_len,
                                  padding_side="right")
         
        special_tokens_dict = dict()
        if tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        # if tokenizer.eos_token is None:
        #     special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        # if tokenizer.bos_token is None:
        #     special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        # if tokenizer.unk_token is None:
        #     special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
        
        #load model    
        model_config = AutoConfig.from_pretrained(args.model_name_or_path)
        model_split_configs, iszeroth_ls, device_to_stage_ls, stage_to_device_ls = stage_partition(args.model_name_or_path)
        sub_network_ls = generate_subnetworks(args, model_split_configs, iszeroth_ls, len(stage_to_device_ls))

        #load lora config
        target_modules = ["q_proj","v_proj"]   #,"gate_proj","down_proj","up_proj"
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            fan_in_fan_out=False,
            lora_dropout=0.,
            inference_mode=False,
            bias="none",
            task_type="CAUSAL_LM",
        )
        for subnetwork_idx in range(len(sub_network_ls)):
            # LoRA(sub_network_ls[subnetwork_idx], r=args.lora_r, alpha=args.lora_alpha, float16=False)
            sub_network_ls[subnetwork_idx] = get_peft_model(sub_network_ls[subnetwork_idx], lora_config)
            # subnetwork.get_input_embeddings().requires_grad_(False)
            # subnetwork.lm_head.requires_grad_(False)
            # subnetwork.print_trainable_parameters()
            sub_network_ls[subnetwork_idx] = sub_network_ls[subnetwork_idx].to(device)
            
        # for subnetwork in sub_network_ls:
        #     # subnetwork.print_trainable_parameters()
        #     for name, param in subnetwork.named_parameters():
        #         if param.requires_grad:
        #             print("name: {}, param.data: {}".format(name, param.data))
        #     print("*"*50)
        # import sys
        # sys.exit(0)
        print("model loaded....")
        watch_video_memory(device)
                
        #  resize
        print("*"*50)
        print("Before adding, tokenizer length: ",len(tokenizer))
        if tokenizer.pad_token is None:
            for subnetwork_idx in range(len(sub_network_ls)):
                smart_tokenizer_and_embedding_resize(
                    special_tokens_dict=special_tokens_dict,
                    tokenizer=tokenizer,
                    model=sub_network_ls[subnetwork_idx],
                )
        if "llama" in args.model:
            if tokenizer.eos_token is None:
                tokenizer.add_special_tokens(
                    {
                        "eos_token": DEFAULT_EOS_TOKEN
                    }
                )
            if tokenizer.eos_token is None:
                tokenizer.add_special_tokens(
                    {
                        "bos_token": DEFAULT_BOS_TOKEN
                    }
                )
            if tokenizer.eos_token is None:
                tokenizer.add_special_tokens(
                    {
                        "unk_token": DEFAULT_UNK_TOKEN
                    }
                )
            
            # tokenizer.add_special_tokens(
            #     {
            #         "eos_token": DEFAULT_EOS_TOKEN,
            #         "bos_token": DEFAULT_BOS_TOKEN,
            #         "unk_token": DEFAULT_UNK_TOKEN,
            #     }
            # )
        print("*"*50)
        print("After adding, tokenizer length: ",len(tokenizer))
        vocab_size = len(tokenizer)
        train_dataset, data_collator, train_dataloader = make_supervised_data_module(tokenizer=tokenizer,args=args)
    
    # trainer = Trainer(model=network, tokenizer=tokenizer, args=training_args, **data_module)
    # trainer.train()
    llm_finetuning(args, sub_network_ls, train_dataloader, device=device)
