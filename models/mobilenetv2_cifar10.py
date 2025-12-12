import torch
from torchvision.models import mobilenet_v2

def get_mobilenet_v2(num_classes=10, width_mult=1.0):
    model = mobilenet_v2(weights=None, width_mult=width_mult)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    return model

