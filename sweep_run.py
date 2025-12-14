import wandb
import torch
import argparse

from models.mobilenetv2_cifar10 import get_mobilenet_v2
from data.load_cifar10 import get_cifar10
from compression.compress_apply import apply_fake_quant_activations
from compression.weights_to_quant import quantize_weights
from compression.quantize import dequantize_symmetric
from train.train import evaluate
import torch.nn as nn
import pickle
import os

def main():
    wandb.init()
    cfg = wandb.config

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    _, testloader = get_cifar10(batch_size=128)

    # Load model
    model = get_mobilenet_v2().to(device)
    base_ckpt = torch.load("checkpoints/mobilenetv2_baseline.pth", map_location="cpu")
    model.load_state_dict(base_ckpt)

    # Quantize weights
    qdict, meta = quantize_weights(base_ckpt, num_bits=cfg.weight_bits)
    model = load_quantized_weights(model, qdict, meta)

    # Quantize activations
    model = apply_fake_quant_activations(model, num_bits=cfg.act_bits)

    # Evaluate
    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc = evaluate(model, testloader, criterion, device)

    # Model size (MB)
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = (total_params * cfg.weight_bits / 8) / (1024 * 1024)

    # Log to WandB
    wandb.log({
        "weight_bits": cfg.weight_bits,
        "act_bits": cfg.act_bits,
        "val_acc": val_acc * 100,
        "val_loss": val_loss,
        "compression_ratio": 32 / cfg.weight_bits,
        "model_size_mb": model_size_mb
    })

    print(
        f"W={cfg.weight_bits}, A={cfg.act_bits} | "
        f"Acc={val_acc*100:.2f}% | "
        f"Model Size={model_size_mb:.2f} MB"
    )

    wandb.finish()

if __name__ == "__main__":
    main()
