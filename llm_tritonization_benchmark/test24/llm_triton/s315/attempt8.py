import triton
import triton.language as tl
import torch

def s315_triton(a):
    # Use PyTorch for argmax reduction as recommended
    max_val = torch.max(a)
    max_idx = torch.argmax(a)
    
    # Return exactly what the C code returns: index + x + 1
    result = max_idx + max_val + 1
    return result