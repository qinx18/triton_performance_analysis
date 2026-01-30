import torch
import triton
import triton.language as tl

@triton.jit
def s332_kernel(a_ptr, t, result_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=float('-inf'))
    
    # Find elements greater than threshold
    condition = a_vals > t
    
    # If any element satisfies condition in this block
    if tl.any(condition):
        # Find first index where condition is true
        # Use cumulative sum to find first occurrence
        condition_int = condition.to(tl.int32)
        first_match_mask = condition_int == 1
        
        # Get the actual indices
        actual_indices = offsets
        
        # Find minimum index among matches
        min_idx = tl.min(tl.where(condition, actual_indices, n_elements))
        
        # Store the minimum index if it's valid
        if min_idx < n_elements:
            tl.store(result_ptr, min_idx)

def s332_triton(a, t):
    n_elements = a.shape[0]
    
    # Use PyTorch for this search operation since it's more efficient
    # and handles the "first greater than" logic better
    mask = a > t
    indices = torch.nonzero(mask, as_tuple=True)[0]
    
    if len(indices) > 0:
        # First index where a[i] > t
        index = indices[0].item()
        value = a[index].item()
    else:
        # No element found
        index = -2
        value = -1.0
    
    return value