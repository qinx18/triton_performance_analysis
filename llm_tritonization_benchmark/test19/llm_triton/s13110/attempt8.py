import torch
import triton
import triton.language as tl

def s13110_triton(aa, len_2d):
    # Use torch.argmax to find the maximum value and its index
    aa_flat = aa.view(-1)
    max_val = torch.max(aa_flat)
    max_idx = torch.argmax(aa_flat)
    
    # Convert flat index back to 2D coordinates
    xindex = max_idx // aa.shape[1]
    yindex = max_idx % aa.shape[1]
    
    # Return max + xindex+1 + yindex+1 (matching C code exactly)
    return max_val + xindex + 1 + yindex + 1