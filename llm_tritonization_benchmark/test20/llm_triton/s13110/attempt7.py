import triton
import triton.language as tl
import torch

def s13110_triton(aa, len_2d):
    N = aa.shape[0]
    
    # Use PyTorch for finding maximum and its indices
    aa_flat = aa.flatten()
    max_val = torch.max(aa_flat)
    max_idx = torch.argmax(aa_flat)
    xindex = max_idx // N
    yindex = max_idx % N
    
    # Return max + (xindex+1) + (yindex+1) as required by C code
    return max_val + (xindex + 1).float() + (yindex + 1).float()