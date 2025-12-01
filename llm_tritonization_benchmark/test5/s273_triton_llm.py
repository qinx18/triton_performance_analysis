import torch
import triton
import triton.language as tl

@triton.jit
def s273_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and compute offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for boundary handling
    mask = offsets < n_elements
    
    # Load data with masking
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Compute d[i] * e[i] once for reuse
    de_product = d_vals * e_vals
    
    # a[i] += d[i] * e[i]
    a_new = a_vals + de_product
    
    # if (a[i] < 0.) b[i] += d[i] * e[i]
    condition_mask = a_new < 0.0
    b_new = b_vals + tl.where(condition_mask, de_product, 0.0)
    
    # c[i] += a[i] * d[i]
    c_new = c_vals + a_new * d_vals
    
    # Store results with masking
    tl.store(a_ptr + offsets, a_new, mask=mask)
    tl.store(b_ptr + offsets, b_new, mask=mask)
    tl.store(c_ptr + offsets, c_new, mask=mask)

def s273_triton(a, b, c, d, e):
    """
    Triton implementation of TSVC s273.
    Optimized with block-level parallelization and memory coalescing.
    """
    # Ensure tensors are contiguous for optimal memory access
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()
    
    n_elements = a.numel()
    
    # Use block size of 1024 for good occupancy
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel with fused operations for better performance
    s273_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a, b, c