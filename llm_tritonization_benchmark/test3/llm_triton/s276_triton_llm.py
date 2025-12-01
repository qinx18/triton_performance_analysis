import torch
import triton
import triton.language as tl

@triton.jit
def s276_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements, mid,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and compute offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load data with masking
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    d = tl.load(d_ptr + offsets, mask=mask)
    
    # Conditional logic: if (i+1 < mid) use c, else use d
    # Note: offsets are 0-based, so condition is (offsets + 1) < mid
    condition = (offsets + 1) < mid
    multiplier = tl.where(condition, c, d)
    
    # Compute update and apply to a
    update = b * multiplier
    result = a + update
    
    # Store result back to a
    tl.store(a_ptr + offsets, result, mask=mask)

def s276_triton(a, b, c, d, mid):
    """
    Triton implementation of TSVC s276 - conditional linear combination
    Uses vectorized conditional operations and optimized memory access patterns
    """
    # Ensure tensors are contiguous for optimal memory access
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    
    n_elements = a.numel()
    
    # Choose block size for optimal occupancy
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s276_kernel[grid](
        a, b, c, d,
        n_elements, mid,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a