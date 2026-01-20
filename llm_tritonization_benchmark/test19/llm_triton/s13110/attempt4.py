import torch
import triton
import triton.language as tl

def s13110_triton(aa):
    N = aa.shape[0]
    
    # Use PyTorch for finding max and its indices
    aa_flat = aa.flatten()
    max_val = torch.max(aa_flat)
    flat_idx = torch.argmax(aa_flat)
    
    xindex = flat_idx // N
    yindex = flat_idx % N
    
    return max_val + (xindex + 1) + (yindex + 1)