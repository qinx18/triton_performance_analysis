import torch
import triton
import triton.language as tl

def s13110_triton(aa):
    N = aa.shape[0]
    
    # Use PyTorch for argmax reduction
    flat_aa = aa.flatten()
    max_val = torch.max(flat_aa).item()
    flat_idx = torch.argmax(flat_aa).item()
    xindex = flat_idx // N
    yindex = flat_idx % N
    
    # Return value matches C code: max + xindex+1 + yindex+1
    return max_val + xindex + 1 + yindex + 1