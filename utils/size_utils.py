import torch

def model_size_bytes(model):
    return sum(p.numel() * p.element_size() for p in model.parameters())

def params_count(model):
    return sum(p.numel() for p in model.parameters())

