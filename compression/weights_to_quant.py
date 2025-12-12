import torch, pickle, argparse, os
from compression.quantize import symmetric_quantize_tensor

def quantize_weights(state_dict, num_bits=8, per_channel=True):
    quantized, metadata = {}, {}

    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.dim() in (4, 2):
            q, scale = symmetric_quantize_tensor(v, num_bits=num_bits, per_channel=per_channel)
            quantized[k] = q.cpu()
            metadata[k + "_scale"] = scale.cpu()
            metadata[k + "_shape"] = v.shape
        else:
            quantized[k] = v.cpu()

    return quantized, metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--ckpt", type=str, default="checkpoints/mobilenetv2_baseline.pth")
    args = parser.parse_args()

    sd = torch.load(args.ckpt, map_location="cpu")
    qdict, meta = quantize_weights(sd, num_bits=args.bits)

    os.makedirs("checkpoints", exist_ok=True)
    out_path = f"checkpoints/weights_quant_{args.bits}bit.pkl"

    with open(out_path, "wb") as f:
        pickle.dump((qdict, meta), f)

    print("Saved:", out_path)

