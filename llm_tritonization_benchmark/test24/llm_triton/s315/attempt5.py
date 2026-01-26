import torch
import triton
import triton.language as tl

def s315_triton(a):
    max_val = torch.max(a)
    max_idx = torch.argmax(a)
    
    result = max_idx + max_val + 1
    return result