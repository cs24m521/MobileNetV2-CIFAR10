import torch
from collections import OrderedDict
import torch.nn as nn

class ActivationCollector:
    def __init__(self):
        self.activations = OrderedDict()

    def hook_fn(self, name):
        def fn(module, input, output):
            self.activations[name] = output.detach().cpu()
        return fn

def collect_activations(model, loader, device, max_batches=1):
    collector = ActivationCollector()
    handles = []

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ReLU, nn.ReLU6)):
            handles.append(module.register_forward_hook(collector.hook_fn(name)))

    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            x = x.to(device)
            model(x)
            if i + 1 >= max_batches:
                break

    for h in handles:
        h.remove()

    return collector.activations

