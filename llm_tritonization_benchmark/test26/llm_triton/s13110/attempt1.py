import torch
import triton
import triton.language as tl

def s13110_triton(aa):
    # Use PyTorch for argmax reduction
    max_val = torch.max(aa)
    flat_idx = torch.argmax(aa.flatten())
    xindex = flat_idx // aa.shape[1]
    yindex = flat_idx % aa.shape[1]
    
    return max_val + xindex + 1 + yindex + 1