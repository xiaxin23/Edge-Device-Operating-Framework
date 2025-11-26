import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, MSELoss
import torch.nn.functional as F
import time

def fixed_cross_entropy(source, target, num_items_in_batch: int = None, ignore_index: int = -100):
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss

def ForCausalLMLoss(
    logits, labels, vocab_size: int, num_items_in_batch: int = None, ignore_index: int = -100
):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    # logits = logits.float()
    if labels.device != logits.device:
        labels = labels.to(logits.device)
    # Shift so that tokens < n predict n
    labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    # shift_labels = shift_labels.to(logits.device)
    # peak = torch.cuda.max_memory_reserved(logits.device)
    # print(f"[before CrossEntropy peak] {peak/1024**3:.2f} GB")
    loss = fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index)
    # loss_eval = loss.detach().item()
    # # peak = torch.cuda.max_memory_reserved(logits.device)
    # # print(f"[CrossEntropy peak] {peak/1024**3:.2f} GB")
    # del logits, shift_labels, loss
    # torch.cuda.empty_cache()
    return loss

def ForSequenceClassificationLoss(labels, pooled_logits, config, **kwargs):
    num_labels = config.num_labels
    if config.problem_type is None:
        if num_labels == 1:
            config.problem_type = "regression"
        elif num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
            config.problem_type = "single_label_classification"
        else:
            config.problem_type = "multi_label_classification"

    labels = labels.to(pooled_logits.device)
    if config.problem_type == "regression":
        loss_fct = MSELoss()
        if num_labels == 1:
            loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
        else:
            loss = loss_fct(pooled_logits, labels)
    elif config.problem_type == "single_label_classification":
        loss = fixed_cross_entropy(pooled_logits.view(-1, num_labels), labels.view(-1), **kwargs)
    elif config.problem_type == "multi_label_classification":
        loss_fct = BCEWithLogitsLoss()
        loss = loss_fct(pooled_logits, labels)
    return loss

def ForCausalLMLoss_chunked(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    ignore_index: int = -100,
    chunk_size: int = 1024
):

    if labels.device != logits.device:
        labels = labels.to(logits.device)

    labels = F.pad(labels, (0, 1), value=ignore_index)      # (B, L+1)
    flat_labels = labels[..., 1:].contiguous().view(-1)     # (N,)

    flat_logits = logits.view(-1, vocab_size)              # (N, V)

    total_loss = torch.tensor(0.0,dtype=torch.float16)
    total_tokens = 0

    N = flat_logits.size(0)
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        l_chunk = flat_logits[i:end]       # (C, V)
        t_chunk = flat_labels[i:end]       # (C,)

        loss_chunk = F.cross_entropy(
            l_chunk, t_chunk,
            ignore_index=ignore_index,
            reduction="sum"
        )
        total_loss += loss_chunk
        total_tokens += int((t_chunk != ignore_index).sum())
        # del l_chunk, t_chunk, loss_chunk
        # torch.cuda.empty_cache()

    mean_loss = total_loss / total_tokens
    # del flat_logits, flat_labels, total_loss
    # torch.cuda.empty_cache()
    return mean_loss

def test_case():
    device = torch.device('cpu')
    B, L, V = 8, 512, 128000  # batch, seq_len, vocab
    logits = torch.randn(B, L, V, device=device, dtype=torch.float16)
    labels = torch.randint(0, V, (B, L), device=device)

    # Warm-up
    _ = ForCausalLMLoss(logits, labels, V)
    _ = ForCausalLMLoss_chunked(logits, labels, V)
    # torch.synchronize()

    n_runs = 10

    # Measure ForCausalLMLoss
    # torch.cuda.reset_peak_memory_stats(device)
    start = time.time()
    for _ in range(n_runs):
        loss1 = ForCausalLMLoss(logits, labels, V)
    # torch.synchronize()
    t1 = (time.time() - start) / n_runs
    # m1 = torch.cuda.max_memory_reserved(device) / (1024**3)

    # Measure chunked version
    # torch.cuda.reset_peak_memory_stats(device)
    start = time.time()
    for _ in range(n_runs):
        loss2 = ForCausalLMLoss_chunked(logits, labels, V, chunk_size=1024)
    # torch.synchronize()
    t2 = (time.time() - start) / n_runs
    # m2 = torch.cuda.max_memory_reserved(device) / (1024**3)

    print(f"ForCausalLMLoss average time: {t1*1000:.2f} ms")#, peak memory: {m1:.2f} GB")
    print(f"Chunked version average time: {t2*1000:.2f} ms")#, peak memory: {m2:.2f} GB")
    print(f"Loss values: {loss1}, {loss2}")

# test_case()