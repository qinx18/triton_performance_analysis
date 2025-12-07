import torch
import triton
import triton.language as tl

def s332_triton(a, t):
    # Find first element greater than threshold
    mask = a > t
    
    if torch.any(mask):
        # Find first index where condition is true
        indices = torch.nonzero(mask, as_tuple=False).flatten()
        if len(indices) > 0:
            first_idx = indices[0].item()
            value = a[first_idx].item()
            return value
    
    # No element found greater than threshold
    return -1.0