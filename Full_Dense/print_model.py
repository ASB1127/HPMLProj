import torch
from transformers import AutoModel

model = AutoModel.from_pretrained("distilbert-base-uncased")

for name, param in model.named_parameters():
    print(f"{name:50s}  shape={list(param.shape)}  params={param.numel()}")


print(model)
