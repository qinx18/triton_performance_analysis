import torch
import triton
import triton.language as tl

@triton.jit
def s124_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s124 - conditional array packing with vectorized operations
    """
    # Get program ID and compute block start
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset range for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load input arrays with masking
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    d = tl.load(d_ptr + offsets, mask=mask)
    e = tl.load(e_ptr + offsets, mask=mask)
    
    # Compute d * e product once
    de_product = d * e
    
    # Vectorized conditional: if b > 0, use b + de_product, else c + de_product
    condition = b > 0.0
    result = tl.where(condition, b + de_product, c + de_product)
    
    # Store result to output array
    tl.store(a_ptr + offsets, result, mask=mask)

def s124_triton(a, b, c, d, e):
    """
    Triton implementation of TSVC s124 - conditional array packing
    Optimized with vectorized operations and efficient memory access patterns
    """
    # Ensure contiguous memory layout for optimal access patterns
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()
    
    n_elements = b.numel()
    
    # Use power-of-2 block size for better memory coalescing
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel with optimized block size
    s124_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a