import torch
import triton
import triton.language as tl

def s13110_triton(aa, len_2d):
    # Find the maximum value and its flat index
    flat_aa = aa.flatten()
    max_val = torch.max(flat_aa)
    flat_idx = torch.argmax(flat_aa)
    
    # Convert flat index to 2D indices
    xindex = flat_idx // aa.shape[1]
    yindex = flat_idx % aa.shape[1]
    
    # Return the exact same value as C code: max + xindex+1 + yindex+1
    return max_val + (xindex + 1) + (yindex + 1)