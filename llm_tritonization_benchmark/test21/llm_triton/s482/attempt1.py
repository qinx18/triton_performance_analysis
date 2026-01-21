import triton
import triton.language as tl
import torch

def s482_triton(a, b, c):
    N = a.shape[0]
    
    # PHASE 1: Find the global exit index on CPU
    condition_mask = c > b
    if torch.any(condition_mask):
        exit_idx = torch.argmax(condition_mask.int()).item()
        # Include exit_idx because computation happens BEFORE exit check
        valid_range = exit_idx + 1
        a[:valid_range] += b[:valid_range] * c[:valid_range]
    else:
        # No exit condition triggered, process all elements
        a[:] += b * c