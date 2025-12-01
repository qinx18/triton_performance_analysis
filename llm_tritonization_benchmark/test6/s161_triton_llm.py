import torch
import triton
import triton.language as tl

@triton.jit
def s161_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and compute offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load data for current block
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    c_next_vals = tl.load(c_ptr + offsets + 1, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Create conditional mask: b[i] < 0
    cond_mask = b_vals < 0.0
    
    # Compute both possible outcomes
    # When b[i] >= 0: a[i] = c[i] + d[i] * e[i]
    a_new = c_vals + d_vals * e_vals
    
    # When b[i] < 0: c[i+1] = a[i] + d[i] * d[i]
    c_next_new = a_vals + d_vals * d_vals
    
    # Apply conditional logic using tl.where
    # Update a[i] when condition is false (b[i] >= 0)
    a_result = tl.where(cond_mask, a_vals, a_new)
    
    # Update c[i+1] when condition is true (b[i] < 0)
    c_next_result = tl.where(cond_mask, c_next_new, c_next_vals)
    
    # Store results
    tl.store(a_ptr + offsets, a_result, mask=mask)
    tl.store(c_ptr + offsets + 1, c_next_result, mask=mask)

def s161_triton(a, b, c, d, e):
    """
    Triton implementation of TSVC s161 - conditional assignment with goto logic.
    Optimized for GPU execution with coalesced memory access and vectorization.
    """
    # Ensure tensors are contiguous and on GPU
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()
    
    # Process n-1 elements (excluding last element)
    n = len(a) - 1
    
    # Choose block size for optimal memory coalescing
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    # Launch kernel
    s161_kernel[grid](
        a, b, c, d, e,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a, c