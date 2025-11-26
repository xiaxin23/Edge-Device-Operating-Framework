import os
import torch
from transformers import AutoModelForCausalLM, AutoConfig

# —— 用户配置 —— 
model_name  = "/home/liyan/Llama-3.2-3B/"
save_dir    = "./model_part_layers/llama3b"
os.makedirs(save_dir, exist_ok=True)

# —— 加载模型到 CPU —— 
# （拆分时放在 CPU 上内存占用可控）
config = AutoConfig.from_pretrained(model_name)
model  = AutoModelForCausalLM.from_pretrained(
    model_name,
    from_tf=bool(".ckpt" in model_name),
    config=config,
    ignore_mismatched_sizes=True,
    # cache_dir="./cache",
    torch_dtype=torch.bfloat16
).to("cpu")

for name, param in model.named_parameters():
    print(name)
state_dict = model.state_dict()

# —— 1. 保存 Embedding 部分 —— 
embed_keys = [k for k in state_dict if k.startswith("model.embed_tokens.")]
embed_dict = {k: state_dict[k] for k in embed_keys}
torch.save(embed_dict, os.path.join(save_dir, "embedding.pt"))
print(f"Saved embedding: {len(embed_keys)} tensors")

# —— 2. 保存每一层 Decoder Layer —— 
num_layers = config.num_hidden_layers
for idx in range(num_layers):
    prefix = f"model.layers.{idx}."
    layer_keys = [k for k in state_dict if k.startswith(prefix)]
    # strip off the prefix so we can load into a fresh LlamaDecoderLayer
    layer_dict = {k[len(prefix):]: state_dict[k] for k in layer_keys}
    torch.save(layer_dict, os.path.join(save_dir, f"layer_{idx}.pt"))
    print(f"Saved layer {idx}: {len(layer_keys)} tensors")

# —— 3. 保存 Head（norm + lm_head）—— 
head_keys = [k for k in state_dict if k.startswith("model.norm.")]
head_dict = {k[len("model.norm."):]: state_dict[k] for k in head_keys}
torch.save(head_dict, os.path.join(save_dir, "head_norm.pt"))
print("Saved head norm:", head_keys)

print("拆分完成，文件保存在", save_dir)
