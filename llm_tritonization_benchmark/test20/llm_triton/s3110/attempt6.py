import torch
import triton
import triton.language as tl

def s3110_triton(aa):
    # Find the maximum value and its indices using PyTorch
    flat_aa = aa.flatten()
    max_val = torch.max(flat_aa)
    max_idx = torch.argmax(flat_aa)
    
    # Convert flat index to 2D coordinates
    xindex = max_idx // aa.shape[1]
    yindex = max_idx % aa.shape[1]
    
    # Calculate chksum (though not used in return)
    chksum = max_val + xindex.float() + yindex.float()
    
    # Return the same value as C code: max + (xindex+1) + (yindex+1)
    return max_val + (xindex + 1).float() + (yindex + 1).float()