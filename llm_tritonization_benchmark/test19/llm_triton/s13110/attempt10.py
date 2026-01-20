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
    
    # Convert to scalar values and ensure they are Python floats
    max_val_scalar = float(max_val.item())
    xindex_scalar = int(xindex.item())
    yindex_scalar = int(yindex.item())
    
    # Return value matches C code: max + xindex+1 + yindex+1
    return max_val_scalar + float(xindex_scalar + 1) + float(yindex_scalar + 1)