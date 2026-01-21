import torch
import triton
import triton.language as tl

def s13110_triton(aa, len_2d):
    # Use PyTorch for global argmax reduction
    flat_aa = aa.flatten()
    max_val = torch.max(flat_aa)
    max_idx = torch.argmax(flat_aa)
    
    xindex = max_idx // len_2d
    yindex = max_idx % len_2d
    
    return max_val + xindex + 1 + yindex + 1