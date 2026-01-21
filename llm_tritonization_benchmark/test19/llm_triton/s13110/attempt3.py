import torch
import triton
import triton.language as tl

def s13110_triton(aa, len_2d):
    # Find the maximum value and its indices using PyTorch
    flat_aa = aa.flatten()
    max_idx = torch.argmax(flat_aa)
    max_val = flat_aa[max_idx]
    
    # Convert flat index to 2D coordinates
    xindex = max_idx // len_2d
    yindex = max_idx % len_2d
    
    # Return the same value as C code: max + xindex+1 + yindex+1
    return max_val + xindex + 1 + yindex + 1