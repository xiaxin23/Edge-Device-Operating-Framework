from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForCausalLM,
    PreTrainedTokenizer, PreTrainedTokenizerFast
)
import os, shutil, json
import torch
from parse_args import parse_args

save_dir = "../resized_model_artifacts"
os.makedirs(save_dir, exist_ok=True)

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

args = parse_args()

# 1) 加载 tokenizer / model
#    建议优先用 fast；若模型不支持 fast，会自动回退到 slow
tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path,
    model_max_length=args.max_seq_len,
    padding_side="right",
    use_fast=True,   # ⭐ 尽量保存出 tokenizer.json
)

config = AutoConfig.from_pretrained(args.model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    config=config,
    ignore_mismatched_sizes=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

# 2) 扩展特殊 tokens 并 resize
special_tokens_dict = {}
if tokenizer.pad_token is None:
    special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
if tokenizer.eos_token is None:
    special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
if tokenizer.bos_token is None:
    special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
if tokenizer.unk_token is None:
    special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

num_added = tokenizer.add_special_tokens(special_tokens_dict)
if num_added > 0:
    model.resize_token_embeddings(len(tokenizer))

# 同步 config 关键字段
model.config.vocab_size   = len(tokenizer)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id
try:
    model.tie_weights()
except Exception:
    pass

# 3) 保存 tokenizer（Fast + Slow 双保险）
#    - Fast: tokenizer.json, tokenizer_config.json, special_tokens_map.json
#    - Slow: vocab.json/merges.txt（BPE）或 spiece.model（SentencePiece）
tokenizer.save_pretrained(save_dir)

# 对 Fast tokenizer，尽量把 slow 资源也保存出来（如果可用）
if isinstance(tokenizer, PreTrainedTokenizerFast) and hasattr(tokenizer, "slow_tokenizer") and tokenizer.slow_tokenizer is not None:
    try:
        tokenizer.slow_tokenizer.save_pretrained(save_dir)
    except Exception:
        pass

# 兜底：拷贝底层资源文件（不同 tokenizer 类的属性名可能不同）
def _maybe_copy(pathlike, target_dir):
    if isinstance(pathlike, str) and os.path.isfile(pathlike):
        dst = os.path.join(target_dir, os.path.basename(pathlike))
        if not os.path.isfile(dst):
            shutil.copy2(pathlike, dst)

# SentencePiece
for attr in ("vocab_file", "sp_model_file", "sp_model", "model_file"):
    _maybe_copy(getattr(tokenizer, attr, None), save_dir)

# BPE（GPT-2/LLamaTokenizer 等 slow 资源）
for attr in ("vocab_file", "merges_file"):
    _maybe_copy(getattr(getattr(tokenizer, "slow_tokenizer", tokenizer), attr, None), save_dir)

# 额外写一次 tokenizer_config.json（确保关键字段齐全）
tok_cfg_path = os.path.join(save_dir, "tokenizer_config.json")
try:
    cfg_json = {}
    if os.path.isfile(tok_cfg_path):
        with open(tok_cfg_path, "r", encoding="utf-8") as f:
            cfg_json = json.load(f)
    cfg_json.setdefault("model_max_length", int(args.max_seq_len))
    cfg_json.setdefault("padding_side", "right")
    cfg_json["unk_token"] = tokenizer.unk_token
    cfg_json["bos_token"] = tokenizer.bos_token
    cfg_json["eos_token"] = tokenizer.eos_token
    cfg_json["pad_token"] = tokenizer.pad_token
    with open(tok_cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg_json, f, ensure_ascii=False, indent=2)
except Exception:
    pass

# 4) 保存模型（safetensors + 分片）
try:
    model.generation_config.save_pretrained(save_dir)
except Exception:
    pass

model.save_pretrained(
    save_dir,
    safe_serialization=True,
    max_shard_size="10GB"
)

print(f"[SAVE OK] artifacts -> {save_dir}")

# 5) 落盘后立即回读自检（强烈建议保留）
from transformers import AutoTokenizer as _AT, AutoModelForCausalLM as _AM, AutoConfig as _AC
tok2 = _AT.from_pretrained(save_dir, use_fast=False, local_files_only=True)
cfg2 = _AC.from_pretrained(save_dir, local_files_only=True)
mdl2 = _AM.from_pretrained(save_dir, torch_dtype="auto", low_cpu_mem_usage=True, local_files_only=True)

emb = mdl2.get_input_embeddings()
print("[SELF-TEST] tokenizer.vocab_size:", tok2.vocab_size, "embed.num_embeddings:", emb.num_embeddings)
assert tok2.vocab_size == emb.num_embeddings, "自检失败：tokenizer 与 embedding 维度不一致！"
print("[SELF-TEST] passed.")


    