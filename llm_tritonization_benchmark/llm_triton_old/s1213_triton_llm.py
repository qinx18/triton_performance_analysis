import torch
import triton
import triton.language as tl

@triton.jit
def s1213_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and calculate offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements (1 to n_elements-2)
    mask = (offsets >= 1) & (offsets < n_elements - 1)
    
    # Load b[i-1] values (with offset adjustment)
    b_prev_offsets = offsets - 1
    b_prev_mask = mask & (b_prev_offsets >= 0)
    b_prev = tl.load(b_ptr + b_prev_offsets, mask=b_prev_mask, other=0.0)
    
    # Load c[i] values
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    
    # Load original a[i+1] values before modification
    a_next_offsets = offsets + 1
    a_next_mask = mask & (a_next_offsets < n_elements)
    a_next = tl.load(a_ptr + a_next_offsets, mask=a_next_mask, other=0.0)
    
    # Load d[i] values
    d_vals = tl.load(d_ptr + offsets, mask=mask, other=0.0)
    
    # Compute new values
    # a[i] = b[i-1] + c[i]
    new_a = b_prev + c_vals
    # b[i] = a[i+1] * d[i] (using original a[i+1])
    new_b = a_next * d_vals
    
    # Store results
    tl.store(a_ptr + offsets, new_a, mask=mask)
    tl.store(b_ptr + offsets, new_b, mask=mask)

def s1213_triton(a, b, c, d):
    """
    Triton implementation of TSVC s1213
    
    Key optimizations:
    - Single kernel launch processes all elements in parallel
    - Coalesced memory access patterns
    - Pre-load original a[i+1] values to handle dependency
    - Efficient masking for boundary conditions
    """
    # Ensure tensors are contiguous
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    
    n_elements = a.shape[0]
    
    # Choose block size for optimal occupancy
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s1213_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a, b