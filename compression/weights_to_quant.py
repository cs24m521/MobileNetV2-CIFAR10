import torch, pickle, argparse, os
from compression.quantize import symmetric_quantize_tensor

def quantize_weights(state_dict, num_bits=8):
    quantized = {}
    metadata = {}

    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1

    for k, v in state_dict.items():

        # Skiping non-weight tensors
        if not isinstance(v, torch.Tensor):
            quantized[k] = v
            continue

        # Skiping biases & BatchNorm params
        if "bias" in k or "bn" in k:
            quantized[k] = v.cpu()
            continue

        w = v.cpu()

        # -------- Conv2d / Depthwise Conv --------
        if w.ndim == 4:
            # per-output-channel quantization
            C = w.size(0)
            max_vals = w.abs().amax(dim=(1, 2, 3))  # shape: [C]
            scales = max_vals / qmax
            scales = torch.clamp(scales, min=1e-8)

            scales_bc = scales.view(C, 1, 1, 1)  # IMPORTANT FIX
            q = torch.round(w / scales_bc).clamp(qmin, qmax).to(torch.int32)

            quantized[k] = q
            metadata[k + "_scale"] = scales
            metadata[k + "_shape"] = w.shape

        # -------- Linear layer --------
        elif w.ndim == 2:
            max_val = w.abs().max()
            scale = max(max_val / qmax, 1e-8)

            q = torch.round(w / scale).clamp(qmin, qmax).to(torch.int32)

            quantized[k] = q
            metadata[k + "_scale"] = scale
            metadata[k + "_shape"] = w.shape

        else:
            quantized[k] = w

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

