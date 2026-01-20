import torch
import triton
import triton.language as tl

@triton.jit
def s332_kernel(a_ptr, t, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=float('-inf'))
    
    # Find elements greater than threshold
    valid_mask = a_vals > t
    combined_mask = mask & valid_mask
    
    # If any valid elements found, get the first one
    if tl.sum(combined_mask.to(tl.int32)) > 0:
        # Create position array for finding first occurrence
        positions = tl.arange(0, BLOCK_SIZE)
        
        # Set invalid positions to a large number
        masked_positions = tl.where(combined_mask, positions, n + 1)
        
        # Find minimum position (first occurrence)
        min_pos = tl.min(masked_positions)
        
        # Get the value at that position
        found_value = tl.load(a_ptr + min_pos)
        
        tl.store(tl.program_id(0) * 2, min_pos)
        tl.store(tl.program_id(0) * 2 + 1, found_value)
    else:
        tl.store(tl.program_id(0) * 2, -2)
        tl.store(tl.program_id(0) * 2 + 1, -1.0)

def s332_triton(a, t):
    n = a.shape[0]
    BLOCK_SIZE = 1024
    
    # Use PyTorch operations for efficiency
    mask = a > t
    if torch.any(mask):
        # Find first index where condition is true
        indices = torch.arange(n, device=a.device)
        valid_indices = torch.where(mask, indices, n)
        index = torch.min(valid_indices).item()
        value = a[index].item()
    else:
        index = -2
        value = -1.0
    
    return value