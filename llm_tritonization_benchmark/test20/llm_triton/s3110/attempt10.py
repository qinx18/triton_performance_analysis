import torch
import triton
import triton.language as tl

def s3110_triton(aa):
    # Use PyTorch for argmax reduction
    flat_tensor = aa.flatten()
    max_val = torch.max(flat_tensor)
    flat_idx = torch.argmax(flat_tensor)
    
    N = aa.shape[0]
    xindex = flat_idx // N
    yindex = flat_idx % N
    
    # Return max + xindex+1 + yindex+1 (as in C code)
    return max_val + (xindex + 1) + (yindex + 1)