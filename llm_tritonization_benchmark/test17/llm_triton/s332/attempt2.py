import torch
import triton
import triton.language as tl

def s332_triton(a, t):
    # Find first value greater than threshold using PyTorch
    mask = a > t
    if torch.any(mask):
        indices = torch.where(mask)[0]
        first_index = indices[0]
        value = a[first_index].item()
    else:
        value = -1.0
    
    return value