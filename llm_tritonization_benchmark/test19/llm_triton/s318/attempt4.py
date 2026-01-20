import triton
import triton.language as tl
import torch

def s318_triton(a, inc):
    n = a.shape[0]
    
    if n == 0:
        return torch.tensor(1.0, device=a.device, dtype=a.dtype)
    
    # Handle strided access
    k = 0
    max_val = torch.abs(a[0])
    index = 0
    
    k += inc
    for i in range(1, n):
        if k >= n:
            break
        abs_ak = torch.abs(a[k])
        if abs_ak > max_val:
            index = i
            max_val = abs_ak
        k += inc
    
    return max_val + index + 1