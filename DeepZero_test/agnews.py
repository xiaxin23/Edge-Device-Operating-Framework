# -*- coding: utf-8 -*-
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler
from transformers.data import DataCollatorWithPadding
from datasets import load_dataset
import numpy as np
import torch.nn as nn
import math
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ============== Logging Setup ==============
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

# ============== LoRA Definition ==============
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

# ============== Hyperparameter Configuration ==============
epochs = 1
batch_size = 16
learning_rate = 5e-5
lora_r = 8
lora_alpha = 16
max_length = 64
log.info(f"batch_size: {batch_size}")
log.info(f"learning_rate: {learning_rate}")
log.info(f"lora_r: {lora_r}")
log.info(f"lora_alpha: {lora_alpha}")
log.info(f"max_length: {max_length}")

model_id = "../Llama-3.2-3B_old"   # Local model path
dataset_path = "./ag_news"         # Local dataset path

id2label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
label2id = {v: k for k, v in id2label.items()}
ds = load_dataset(dataset_path)
text_name = "text"
test_name = "test"

# ============== Tokenizer & Model ==============
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
)
model.config.pad_token_id = tokenizer.pad_token_id

# Inject LoRA
replace_linear_with_lora(model, lora_r, lora_alpha)
for n, p in model.named_parameters():
    if "lora" not in n:
        p.requires_grad = False

# ============== Data Preprocessing ==============
def preprocess_function(examples):
    tokenized = tokenizer(examples[text_name], max_length=max_length, truncation=True)
    tokenized["labels"] = examples["label"]
    return tokenized

tokenized_ds = ds.map(preprocess_function, batched=True, remove_columns=[text_name])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_loader = DataLoader(
    tokenized_ds["train"], batch_size=batch_size, shuffle=True, collate_fn=data_collator
)
eval_loader = DataLoader(
    tokenized_ds[test_name], batch_size=batch_size, shuffle=False, collate_fn=data_collator
)

log.info(f"Training set size: {len(tokenized_ds['train'])} samples")
log.info(f"Test set size: {len(tokenized_ds[test_name])} samples")

# ============== Optimizer & Scheduler ==============
optimizer = AdamW(model.parameters(), lr=learning_rate)
num_training_steps = epochs * len(train_loader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ============== Evaluation Function ==============
def evaluate(model, dataloader, device, desc="Evaluating"):
    model.eval()
    preds, labels = [], []
    eval_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            eval_loss += outputs.loss.item()

            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=-1)
            preds.extend(batch_preds.cpu().numpy())
            labels.extend(batch["labels"].cpu().numpy())

    # ====== Compute Metrics ======
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    avg_eval_loss = eval_loss / len(dataloader)

    log.info(f"[{desc}] Loss={avg_eval_loss:.4f}, "
             f"Acc={acc:.4f}, Precision={precision:.4f}, "
             f"Recall={recall:.4f}, F1={f1:.4f}")

    model.train()

# ============== Evaluate Once Before Training ==============
evaluate(model, eval_loader, device, desc="Initial Evaluation")

# ============== Training Loop ==============
model.train()
total_loss = 0

for step, batch in enumerate(train_loader):
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch)
    loss = outputs.loss

    # Backward
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    total_loss += loss.item()
    log.info(f"Step {step+1} - Loss: {loss.item():.4f}")

    # Evaluate every 100 steps
    if (step + 1) % 2 == 0:
        evaluate(model, eval_loader, device, desc=f"Step {step+1} Evaluation")
