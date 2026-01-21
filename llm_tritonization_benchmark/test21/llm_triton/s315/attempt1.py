import torch
import triton
import triton.language as tl

def s315_triton(a):
    # Use PyTorch's built-in functions for argmax reduction
    max_val = torch.max(a)
    max_idx = torch.argmax(a)
    
    # Return exactly what the C code returns: index + x + 1
    return max_idx + max_val + 1