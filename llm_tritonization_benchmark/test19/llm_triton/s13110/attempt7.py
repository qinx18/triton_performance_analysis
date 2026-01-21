import torch
import triton
import triton.language as tl

def s13110_triton(aa, len_2d):
    # Flatten the 2D array to use torch.argmax
    aa_flat = aa.flatten()
    
    # Find max value and its index
    max_val = torch.max(aa_flat)
    max_idx = torch.argmax(aa_flat)
    
    # Convert flat index to 2D coordinates
    xindex = max_idx // aa.shape[1]
    yindex = max_idx % aa.shape[1]
    
    # Return exactly what C code returns: max + xindex+1 + yindex+1
    # Note: C uses 0-based indexing but returns xindex+1, yindex+1
    return max_val + xindex + 1 + yindex + 1