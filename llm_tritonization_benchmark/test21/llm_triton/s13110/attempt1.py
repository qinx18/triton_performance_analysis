import torch
import triton
import triton.language as tl

def s13110_triton(aa, len_2d):
    # This is an argmax reduction - use PyTorch for efficiency
    aa_flat = aa.flatten()
    max_val = torch.max(aa_flat)
    flat_idx = torch.argmax(aa_flat)
    
    xindex = flat_idx // aa.shape[1]
    yindex = flat_idx % aa.shape[1]
    
    # Return matches C code: max + xindex+1 + yindex+1
    return max_val + xindex + 1 + yindex + 1