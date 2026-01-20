import triton
import triton.language as tl
import torch

def s3110_triton(aa):
    N = aa.shape[0]
    
    # Use PyTorch for argmax reduction
    flat_aa = aa.flatten()
    max_val = torch.max(flat_aa)
    max_idx = torch.argmax(flat_aa)
    
    xindex = max_idx // N
    yindex = max_idx % N
    
    # Return max + xindex+1 + yindex+1 as per C code
    return max_val + xindex + 1 + yindex + 1