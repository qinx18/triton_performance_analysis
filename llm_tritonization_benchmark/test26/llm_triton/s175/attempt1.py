import triton
import triton.language as tl
import torch

@triton.jit
def s175_kernel(a_ptr, a_copy_ptr, b_ptr, inc, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    # Apply stride pattern: only process indices that would be hit by i += inc loop
    # Map linear thread indices back to strided indices
    strided_indices = (indices // inc) * inc + (indices % inc)
    
    # Only process indices that are multiples of inc and within bounds
    valid_stride = (strided_indices % inc) == 0
    valid_bounds = strided_indices < (n_elements - 1)
    mask = valid_stride & valid_bounds
    
    # Load values where mask is true
    a_copy_vals = tl.load(a_copy_ptr + strided_indices, mask=mask, other=0.0)
    a_copy_vals_inc = tl.load(a_copy_ptr + strided_indices + inc, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + strided_indices, mask=mask, other=0.0)
    
    # Compute result
    result = a_copy_vals_inc + b_vals
    
    # Store result
    tl.store(a_ptr + strided_indices, result, mask=mask)

def s175_triton(a, b, inc=1):
    n_elements = a.shape[0]
    
    # Create read-only copy to handle WAR dependencies
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    grid = (grid_size,)
    
    s175_kernel[grid](
        a, a_copy, b, inc, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a