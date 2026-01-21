import torch
import triton
import triton.language as tl

def s3110_triton(aa):
    n = aa.shape[0]
    
    # Use PyTorch for argmax (more efficient and reliable)
    flat_aa = aa.flatten()
    max_val = torch.max(flat_aa)
    flat_idx = torch.argmax(flat_aa)
    xindex = flat_idx // n
    yindex = flat_idx % n
    
    return max_val + xindex + 1 + yindex + 1