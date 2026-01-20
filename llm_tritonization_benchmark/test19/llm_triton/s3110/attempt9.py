import triton
import triton.language as tl
import torch

def s3110_triton(aa):
    # Use PyTorch for the 2D argmax reduction
    flat_aa = aa.flatten()
    max_val = torch.max(flat_aa)
    max_idx = torch.argmax(flat_aa)
    
    N = aa.shape[0]
    xindex = max_idx // N
    yindex = max_idx % N
    
    # Return max + xindex+1 + yindex+1 as per C code
    result = max_val + xindex + 1 + yindex + 1
    return float(result.item())