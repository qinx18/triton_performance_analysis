import torch
import triton
import triton.language as tl

def s3110_triton(aa):
    N = aa.shape[0]
    
    # Use PyTorch for argmax reduction
    flat_aa = aa.flatten()
    max_val = torch.max(flat_aa)
    flat_idx = torch.argmax(flat_aa)
    
    xindex = flat_idx // N
    yindex = flat_idx % N
    
    result = max_val + xindex + 1 + yindex + 1
    return result.item()