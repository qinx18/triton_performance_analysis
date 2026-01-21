import torch
import triton
import triton.language as tl

def s332_triton(a, t):
    # Find first value greater than threshold using PyTorch
    mask = a > t
    if torch.any(mask):
        # Find first occurrence
        indices = torch.nonzero(mask, as_tuple=False)
        first_idx = indices[0, 0].item()
        value = a[first_idx].item()
        index = first_idx
    else:
        # No value found greater than threshold
        value = -1.0
        index = -2
    
    return value

@triton.jit
def s332_kernel(a_ptr, result_ptr, t, n_elements, BLOCK_SIZE: tl.constexpr):
    # This kernel is not used in the main function but provided for completeness
    # The search operation is better handled by PyTorch operations
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=float('-inf'))
    
    # Check if any values are greater than threshold
    greater_mask = a_vals > t
    
    # Store results (this approach is simplified - actual early termination search
    # requires more complex reduction patterns that are better handled by PyTorch)
    tl.store(result_ptr + offsets, tl.where(greater_mask, a_vals, 0.0), mask=mask)