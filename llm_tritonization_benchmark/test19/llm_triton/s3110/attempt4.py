import torch
import triton
import triton.language as tl

def s3110_triton(aa):
    N = aa.shape[0]
    
    # Use PyTorch for the argmax reduction
    flat_aa = aa.flatten()
    max_val = torch.max(flat_aa)
    max_idx = torch.argmax(flat_aa)
    
    xindex = max_idx // N
    yindex = max_idx % N
    
    return max_val + (xindex + 1) + (yindex + 1)