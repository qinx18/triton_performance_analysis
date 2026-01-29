import torch
import triton
import triton.language as tl

def s341_triton(a, b):
    n_elements = b.shape[0]
    
    # Use PyTorch's boolean indexing for stream compaction
    mask = b > 0.0
    packed_values = b[mask]
    num_packed = packed_values.numel()
    
    # Pack positive values into the beginning of array a
    a[:num_packed] = packed_values
    
    return a