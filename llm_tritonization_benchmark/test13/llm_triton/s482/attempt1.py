import torch
import triton
import triton.language as tl

def s482_triton(a, b, c):
    n = a.shape[0]
    
    # Find first index where c[i] > b[i]
    condition_mask = c > b
    
    if torch.any(condition_mask):
        exit_idx = torch.argmax(condition_mask.int()).item()
        # Process elements 0 to exit_idx (inclusive, since computation happens before break)
        valid_range = exit_idx + 1
        if valid_range > 0:
            a[:valid_range] += b[:valid_range] * c[:valid_range]
    else:
        # No early exit, process all elements
        a[:] += b * c