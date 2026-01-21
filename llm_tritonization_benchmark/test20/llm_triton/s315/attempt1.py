import torch
import triton
import triton.language as tl

def s315_triton(a):
    # Use PyTorch's argmax for finding the maximum value and its index
    max_val = torch.max(a)
    max_idx = torch.argmax(a)
    
    # Return the same value as C code: index + x + 1
    return max_idx.item() + max_val.item() + 1