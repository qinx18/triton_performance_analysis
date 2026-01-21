import torch
import triton
import triton.language as tl

def s3110_triton(aa):
    # Use PyTorch for argmax reduction
    flat_aa = aa.flatten()
    max_val = torch.max(flat_aa)
    flat_idx = torch.argmax(flat_aa)
    xindex = flat_idx // aa.shape[1]
    yindex = flat_idx % aa.shape[1]
    
    return max_val + xindex + yindex