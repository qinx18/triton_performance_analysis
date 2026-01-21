import triton
import triton.language as tl
import torch

def s481_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    # Check for early exit condition globally first
    exit_mask = d < 0.0
    if torch.any(exit_mask):
        exit_idx = torch.argmax(exit_mask.int()).item()
        # Process only elements before exit point
        if exit_idx > 0:
            a[:exit_idx] += b[:exit_idx] * c[:exit_idx]
        # Exit here - no further processing
        return
    
    # No exit condition, process all elements
    a[:] += b[:] * c[:]