import torch
import triton
import triton.language as tl

@triton.jit
def s2711_kernel(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and compute offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load data with masking
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    
    # Compute conditional update: if b != 0, then a += b * c
    b_nonzero = b_vals != 0.0
    result = tl.where(b_nonzero, a_vals + b_vals * c_vals, a_vals)
    
    # Store result
    tl.store(a_ptr + offsets, result, mask=mask)

def s2711_triton(a, b, c):
    """
    Triton implementation of TSVC s2711
    Performs conditional multiply-add: a[i] += b[i] * c[i] where b[i] != 0
    """
    # Ensure contiguous memory layout for optimal access patterns
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    
    n_elements = a.numel()
    
    # Use power-of-2 block size for memory coalescing
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel with vectorized memory operations
    s2711_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a