import torch, pickle, argparse
from models.mobilenetv2_cifar10 import get_mobilenet_v2
from train.train import evaluate
from compression.compress_apply import apply_fake_quant_activations
from compression.quantize import dequantize_symmetric
from data.load_cifar10 import get_cifar10
import torch.nn as nn

def compute_model_size_mb(model, weight_bits):
    total_params = sum(p.numel() for p in model.parameters())
    size_bytes = total_params * weight_bits / 8
    return size_bytes / (1024 * 1024)

def load_quantized_weights(model, qdict, meta):
    new_state = {}

    for k, v in model.state_dict().items():
        if k in qdict and (k + "_scale") in meta:
            scale = meta[k + "_scale"]
            new_state[k] = dequantize_symmetric(qdict[k], scale)
        else:
            new_state[k] = v

    model.load_state_dict(new_state, strict=False)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_bits", type=int, default=8)
    parser.add_argument("--act_bits", type=int, default=8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainloader, testloader = get_cifar10(batch_size=128)

    model = get_mobilenet_v2().to(device)
    state = torch.load("checkpoints/mobilenetv2_baseline.pth", map_location="cpu")
    model.load_state_dict(state)

    # load quantized weights
    try:
        with open(f"checkpoints/weights_quant_{args.weight_bits}bit.pkl", "rb") as f:
            qdict, meta = pickle.load(f)
        model = load_quantized_weights(model, qdict, meta)
    except:
        print("Warning: quantized weights not found, using FP32.")

    model = apply_fake_quant_activations(model, num_bits=args.act_bits)

    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc = evaluate(model, testloader, criterion, device)
    model_size_mb = compute_model_size_mb(model, args.weight_bits)

    print(f"Accuracy: {val_acc*100:.4f} | Loss: {val_loss:.4f}")

