import torch

def symmetric_quantize_tensor(tensor, num_bits=8, per_channel=False, channel_dim=0):
    qmin = -(2 ** (num_bits - 1))
    qmax = (2 ** (num_bits - 1)) - 1

    if per_channel and tensor.dim() > 1:
        reduce_dims = tuple(i for i in range(tensor.dim()) if i != channel_dim)
        max_vals = tensor.abs().amax(dim=reduce_dims)
        scales = max_vals / float(qmax)
        scales = torch.clamp(scales, min=1e-8)

        q = torch.round(tensor / scales.view([-1] + [1] * (tensor.dim() - 1))).clamp(qmin, qmax).to(torch.int32)
        return q, scales
    else:
        max_val = tensor.abs().max()
        scale = max(max_val / float(qmax), 1e-8)
        q = torch.round(tensor / scale).clamp(qmin, qmax).to(torch.int32)
        return q, scale

def dequantize_symmetric(q, scale):
    if isinstance(scale, torch.Tensor) and q.ndim == 4:
        scale = scale.view(q.size(0), 1, 1, 1)
    return q.float() * scale

def asymmetric_quantize_tensor(tensor, num_bits=8):
    qmin, qmax = 0, 2**num_bits - 1
    min_val, max_val = float(tensor.min()), float(tensor.max())

    if max_val - min_val == 0:
        scale, zero_point = 1.0, 0
    else:
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = int(qmin - min_val / scale)
        zero_point = max(qmin, min(qmax, zero_point))

    q = ((tensor / scale) + zero_point).round().clamp(qmin, qmax).to(torch.uint8)
    return q, scale, zero_point, min_val, max_val

def dequantize_asymmetric(q, scale, zero_point):
    return (q.float() - zero_point) * scale

