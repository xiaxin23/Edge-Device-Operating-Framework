import torch
import torch.nn as nn

def stage_partition(training_method, model_name, devices_ls=None):
    if training_method == "only_zeroth":
        model_split_configs = [(0,6),(6,14),(14,22),(22,28)]
        iszeroth_ls = [True,True,True,True]
        device_to_stage_ls = [0,1,2,3]
        stage_to_device_ls = [0,1,2,3]
    elif training_method == "only_first" or training_method == "pipedream":
        model_split_configs = [(0,3),(3,6),(6,9),(9,16),(16,23),(23,28)]
        iszeroth_ls = [False,False,False,False,False,False]
        device_to_stage_ls = [0,1,2,3,4,5]
        stage_to_device_ls = [0,1,2,3,4,5]
        to_device_ls = ["cpu", "cpu", "cpu", "cuda", "cuda", "cuda"]
    elif training_method == "HDP":
        model_split_configs = [(0,14),(6,28),(0,14),(6,28)]
        iszeroth_ls = [False,False,False,False]
        device_to_stage_ls = [0,1,0,1]
        stage_to_device_ls = [0,1,0,1]
        stage_count_ls = [2,2,2,2]
        stage_count_id_ls = [0,0,1,1]
        return model_split_configs, iszeroth_ls, device_to_stage_ls, stage_to_device_ls, stage_count_ls, stage_count_id_ls
    elif training_method == "HPP":
        model_split_configs = [(0,14),(0,14),(14,22),(22,28)]
        iszeroth_ls = [False,False,False,False]
        device_to_stage_ls = [0,0,1,2]
        stage_to_device_ls = [0,0,1,2]
        stage_count_ls = [2,2,1,1]
        stage_count_id_ls = [0,1,0,0]
        return model_split_configs, iszeroth_ls, device_to_stage_ls, stage_to_device_ls, stage_count_ls, stage_count_id_ls
    elif "hybrid" in training_method:
        model_split_configs = [(0,3),(3,6),(6,9),(9,16),(16,23),(23,28)]
        iszeroth_ls = [True,True,True,False,False,False]
        device_to_stage_ls = [0,1,2,3,4,5]
        stage_to_device_ls = [0,1,2,3,4,5]
        to_device_ls = ["npu", "npu", "npu", "cuda", "cuda", "cuda"]
    else:
        print("training_method error!!")
        import sys
        sys.exit(0)
    return model_split_configs, iszeroth_ls, device_to_stage_ls, stage_to_device_ls, to_device_ls