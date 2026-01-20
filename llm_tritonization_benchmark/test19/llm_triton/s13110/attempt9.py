import torch
import triton
import triton.language as tl

def s13110_triton(aa):
    N = aa.shape[0]
    
    # Use PyTorch for argmax reduction
    flat_aa = aa.flatten()
    max_val = torch.max(flat_aa)
    flat_idx = torch.argmax(flat_aa)
    xindex = flat_idx // N
    yindex = flat_idx % N
    
    # Convert to scalar values to ensure float conversion works
    max_val_scalar = max_val.item()
    xindex_scalar = xindex.item()
    yindex_scalar = yindex.item()
    
    # Return value matches C code: max + xindex+1 + yindex+1
    return max_val_scalar + xindex_scalar + 1 + yindex_scalar + 1