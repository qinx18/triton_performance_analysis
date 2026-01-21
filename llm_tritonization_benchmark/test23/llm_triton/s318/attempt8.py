import triton
import triton.language as tl
import torch

def s318_triton(a, abs, inc):
    n = a.shape[0]
    
    # Initialize with first element
    max_val = torch.abs(a[0])
    max_idx = 0
    
    # Handle stride access manually
    k = inc
    for i in range(1, n):
        if k < n:
            curr_abs = torch.abs(a[k])
            if curr_abs > max_val:
                max_val = curr_abs
                max_idx = i
        k += inc
    
    return max_val + max_idx + 1