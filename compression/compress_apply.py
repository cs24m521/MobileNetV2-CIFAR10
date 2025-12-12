import torch.nn as nn
from compression.quantize import asymmetric_quantize_tensor, dequantize_asymmetric

class FakeQuantAct(nn.Module):
    def __init__(self, num_bits=8):
        super().__init__()
        self.num_bits = num_bits

    def forward(self, x):
        q, scale, zp, mn, mx = asymmetric_quantize_tensor(x, self.num_bits)
        return dequantize_asymmetric(q, scale, zp)

def find_parent_module(model, name):
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]

def apply_fake_quant_activations(model, num_bits=8):
    for name, module in list(model.named_modules()):
        if isinstance(module, (nn.ReLU, nn.ReLU6)):
            parent, attr = find_parent_module(model, name)
            setattr(parent, attr, nn.Sequential(module, FakeQuantAct(num_bits)))
    return model

