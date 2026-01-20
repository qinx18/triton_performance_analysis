import torch
import triton
import triton.language as tl

def s3110_triton(aa):
    # Use PyTorch for efficient argmax computation on flattened array
    flat_aa = aa.flatten()
    max_val = torch.max(flat_aa)
    max_idx = torch.argmax(flat_aa)
    
    # Convert flat index back to 2D coordinates
    N = aa.shape[0]  # Should be same as aa.shape[1] for square matrix
    xindex = max_idx // N
    yindex = max_idx % N
    
    # Return the exact value as C code: max + xindex+1 + yindex+1
    result = max_val + (xindex + 1).float() + (yindex + 1).float()
    
    return result.item()