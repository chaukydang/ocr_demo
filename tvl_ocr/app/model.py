import torch
from transformers import AutoModel, AutoTokenizer

model_name = "5CD-AI/Vintern-1B-v3_5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_model():
    try:
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_flash_attn=False,
        ).eval().to(device)
    except:
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    return model, tokenizer

model, tokenizer = load_model()
