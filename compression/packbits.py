import numpy as np

def pack_bits_array_size(num_elements, bitwidth):
    if bitwidth == 32:
        return num_elements * 4
    if bitwidth == 8:
        return num_elements
    total_bits = num_elements * bitwidth
    return (total_bits + 7) // 8

def compute_weights_storage(qdict, metadata, bitwidth=8):
    total_bytes = 0
    for k, v in qdict.items():
        if hasattr(v, "numpy"):
            arr = v.numpy().ravel()
            if arr.dtype in (np.int32, np.int8, np.uint8):
                total_bytes += pack_bits_array_size(arr.size, bitwidth)
            else:
                total_bytes += arr.nbytes

    # metadata: scales are float32
    for k in metadata:
        if k.endswith("_scale"):
            s = metadata[k]
            try:
                total_bytes += s.numel() * 4
            except:
                total_bytes += 4

    return total_bytes

