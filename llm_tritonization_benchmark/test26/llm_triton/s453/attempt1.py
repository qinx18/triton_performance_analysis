import triton
import triton.language as tl
import torch

@triton.jit
def s453_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Sequential computation: s starts at 0 and accumulates 2.0 for each element
    # At position i: s = (i + 1) * 2.0
    current_offsets = block_start + offsets
    mask = current_offsets < n_elements
    
    # Calculate s values: s = (position + 1) * 2.0
    s_values = (current_offsets + 1) * 2.0
    
    # Load b values
    b_vals = tl.load(b_ptr + current_offsets, mask=mask, other=0.0)
    
    # Compute a[i] = s * b[i]
    a_vals = s_values * b_vals
    
    # Store result
    tl.store(a_ptr + current_offsets, a_vals, mask=mask)

def s453_triton(a, b):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Launch kernel
    s453_kernel[(grid_size,)](
        a, b, n_elements, BLOCK_SIZE
    )
    
    return a