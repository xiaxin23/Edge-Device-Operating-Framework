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

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    
def parse_args():
    parser = argparse.ArgumentParser("Pretraining")
    parser.add_argument('--device', type=int, default=2)
    parser.add_argument('--seed', type=int, default=324823217)
    parser.add_argument('--method', choices=['rge', 'bge', 'cge', 'no'], default='no')
    parser.add_argument('--dataset', choices=['cifar10','mnist'], default='cifar10')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--zo_step_size', type=float, default=0.01)
    parser.add_argument('--q_num', type=int, default=10)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--warm-epochs', type=int, default=3)
    
    args = parser.parse_args()
    return args
    
@torch.no_grad()
def f(params_dict, network, x, y, loss_func):
    state_dict_backup = network.state_dict()
    network.load_state_dict(params_dict, strict=False)
    loss = loss_func(network(x), y).detach().item()
    network.load_state_dict(state_dict_backup)
    return loss

@torch.no_grad()
def rge(func, params_dict, sample_size, step_size, base=None):
    if base == None:
        base = func(params_dict)
    grads_dict = {}
    for _ in range(sample_size):
        perturbs_dict, perturbed_params_dict = {}, {}
        for key, param in params_dict.items():
            perturb = torch.randn_like(param)
            perturb /= (torch.norm(perturb) + 1e-8)
            # perturb *= step_size
            perturbs_dict[key] = perturb
            perturbed_params_dict[key] = step_size * perturb + param
        directional_derivative = (func(perturbed_params_dict) - base) / step_size
        if len(grads_dict.keys()) == len(params_dict.keys()):
            for key, perturb in perturbs_dict.items():
                grads_dict[key] += perturb * directional_derivative / sample_size
        else:
            for key, perturb in perturbs_dict.items():
                grads_dict[key] = perturb * directional_derivative / sample_size
    return grads_dict

@torch.no_grad()
def cge(func, params_dict, step_size, base=None):
    if base == None:
        base = func(params_dict)
    grads_dict = {}
    for key, param in params_dict.items():
        mask_flat = torch.ones_like(param).flatten()
        directional_derivative = torch.zeros_like(param)
        directional_derivative_flat = directional_derivative.flatten()
        for idx in mask_flat.nonzero().flatten():
            perturbed_params_dict = deepcopy(params_dict)
            p_flat = perturbed_params_dict[key].flatten()
            p_flat[idx] += step_size
            directional_derivative_flat[idx] = (func(perturbed_params_dict) - base) / step_size
        grads_dict[key] = directional_derivative.to(param.device)
    return grads_dict

@torch.no_grad()
def bge(func, params_dict, sample_size, step_size, base=None):
    if base == None:
        base = func(params_dict)
    grads_dict = {}
    replace_key_ls = ["linear_a_q", "linear_b_q", "linear_a_v", "linear_b_v"]
    perturbs_dict, perturbed_params_dict = {}, {}
    for key, param in params_dict.items():
        if "linear_a_q" in key:
            for _ in range(sample_size):
                for key_idx in range(len(replace_key_ls)):
                    new_key = key.replace("linear_a_q", replace_key_ls[key_idx])
                    perturb = torch.randn_like(params_dict[new_key])
                    perturb /= (torch.norm(perturb) + 1e-8)  #newnew
                    # perturb *= step_size
                    perturbs_dict[new_key] = perturb
                    perturbed_params_dict[new_key] = step_size * perturb + params_dict[new_key]
                directional_derivative = (func(perturbed_params_dict) - base) / step_size
                for key_idx in range(len(replace_key_ls)):
                    new_key = key.replace("linear_a_q", replace_key_ls[key_idx])
                    if new_key in grads_dict.keys():
                        grads_dict[new_key] += perturbs_dict[new_key] * directional_derivative / sample_size
                    else:
                        grads_dict[new_key] = perturbs_dict[new_key] * directional_derivative / sample_size
    return grads_dict

def zero_train(args, model, train_data, device):
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)
    global_length = (args.epoch - args.warm_epochs) * len(train_data)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=global_length)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    with torch.no_grad():
        for idx, (images,labels) in tqdm(enumerate(train_data)):
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            f_theta = partial(f, network=model, x=images, y=labels, loss_func=criterion)
            params_dict = {
                name: p for name, p in model.named_parameters() if p.requires_grad
                }
            if args.method == "rge":
                grads_dict = rge(f_theta, params_dict, args.q_num, args.lr)
            elif args.method == "cge":
                grads_dict = cge(f_theta, params_dict, args.lr)
            elif args.method == "bge":
                grads_dict = bge(f_theta, params_dict, args.q_num, args.lr)
            for key, param in params_dict.items():
                param.grad = grads_dict[key]
            optimizer.step()
            # break
        
    return model
    
def train(model, train_data, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    # 开始训练 
    model.train()
    for idx, (images,labels) in tqdm(enumerate(train_data)):
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)
        out = model(images)
        predict = torch.argmax(out, dim=-1)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
    return model

def test(model, test_data, device):
    criterion = torch.nn.CrossEntropyLoss().to(device)
    model.eval()
    total_val_loss, total_val_acc = 0, 0
    # cnt = 0
    with torch.no_grad():
        for idx, (images, labels) in tqdm(enumerate(test_data)):
            images = images.to(device)
            labels = labels.to(device)
            out = model(images)
            predict = torch.argmax(out, dim=-1)
            loss = criterion(out, labels)
            equal = torch.eq(predict, labels)
            accuracy = torch.mean(equal.float())
            # print(accuracy)
            total_val_loss += loss.item()
            total_val_acc += accuracy.item()
            # cnt += 1
    print('total_val_loss: {}, total_val_acc: {}'.format(total_val_loss/len(test_data), total_val_acc/len(test_data)))
    
def pretraining(args, model, loaders, device):
    criterion = torch.nn.CrossEntropyLoss().to(device)
    train_data = loaders['train']
    test_data = loaders['test']
    # test(model, test_data, device)
    print(args.method)
    for epoch in range(100):
        model.train()
        print('Epoch: ', epoch)
        if args.method != 'no':
            # print("kkk")
            # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
            # global_length = (args.epoch - args.warm_epochs) * len(train_data)
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=global_length)
            criterion = torch.nn.CrossEntropyLoss().to(device)
            with torch.no_grad():
                for idx, (images,labels) in tqdm(enumerate(train_data)):
                    optimizer.zero_grad()
                    images = images.to(device)
                    labels = labels.to(device)
                    # output = model(input)
                    # loss.backward
                    f_theta = partial(f, network=model, x=images, y=labels, loss_func=criterion)
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
            criterion = torch.nn.CrossEntropyLoss().to(device)
            # 开始训练 
            # model.train()
            for idx, (images,labels) in tqdm(enumerate(train_data)):
                optimizer.zero_grad()
                images = images.to(device)
                labels = labels.to(device)
                out = model(images)
                # predict = torch.argmax(out, dim=-1)
                loss = criterion(out, labels)
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
        test(model, test_data, device)

if __name__ == "__main__":
    args = parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    set_seed(args.seed)    
    device = f"cuda:{args.device}"
    network_init_func = vit_lora   #vit_with_classifiers
    network_kwargs = {'num_classes': 10}
    network = network_init_func(**network_kwargs)
    # time.sleep(100)
    param_dict = torch.load(f"./model_store/vit_model.pth", map_location=torch.device('cpu'))
    # time.sleep(100)
    network.load_state_dict(param_dict, strict=False)
    del param_dict
    # time.sleep(100)
    network = network.to(device)
    # time.sleep(100)
    # for key,param in network.named_parameters():
    #     if not key.startswith('head'):
    #         param.requires_grad = False
    for key,param in network.named_parameters():
        if '_a_' in key or '_b_' in key:
            continue
        else:
            param.requires_grad = False
    # for key,param in network.named_parameters():
    #     if param.requires_grad:
    #         print(key)
    loaders, class_num = prepare_dataset(args.dataset, args.batch_size, device=device)
    pretraining(args, network, loaders, device=device)
    # pretraining(None, network, loaders, device=device)
