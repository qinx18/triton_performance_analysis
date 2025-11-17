import torch
import triton
import triton.language as tl

@triton.jit
def s1281_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s1281: x = b*c + a*d + e; a = x-1; b = x
    Uses coalesced memory access and vectorized operations
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load all input vectors with masking for edge cases
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Compute x = b*c + a*d + e using fused multiply-add operations
    x = b_vals * c_vals + a_vals * d_vals + e_vals
    
    # Compute outputs: a = x - 1.0, b = x
    new_a = x - 1.0
    new_b = x
    
    # Store results with masking
    tl.store(a_ptr + offsets, new_a, mask=mask)
    tl.store(b_ptr + offsets, new_b, mask=mask)

def s1281_triton(a, b, c, d, e):
    """
    Triton implementation of TSVC s1281 function.
    Optimized with block-level parallelism and coalesced memory access.
    """
    # Ensure contiguous memory layout for optimal performance
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()
    
    n_elements = a.numel()
    
    # Choose block size for optimal occupancy (power of 2, multiple of warp size)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel with optimal block size
    s1281_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a, b