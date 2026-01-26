import torch
import triton
import triton.language as tl

def s315_triton(a):
    max_val = torch.max(a)
    max_idx = torch.argmax(a)
    
    result = max_val + max_idx + 1
    return result.item()