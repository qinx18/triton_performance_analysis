import torch
import triton
import triton.language as tl

@triton.jit
def s293_kernel(a_ptr, val, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel that sets all elements of array to a[0] value.
    Uses coalesced memory writes for optimal performance.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to handle array boundaries
    mask = offsets < n_elements
    
    # Broadcast the scalar value to all elements in this block
    vals = tl.full([BLOCK_SIZE], val, dtype=tl.float32)
    
    # Store values with masking for edge cases
    tl.store(a_ptr + offsets, vals, mask=mask)

def s293_triton(a):
    """
    Triton implementation of TSVC s293 function.
    Optimized with coalesced memory access and efficient broadcasting.
    """
    a = a.contiguous()
    n_elements = a.numel()
    
    # Read the broadcast value once
    val = a[0].item()
    
    # Use block size optimized for memory coalescing
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel with optimal grid configuration
    s293_kernel[grid](
        a, val, n_elements, BLOCK_SIZE
    )
    
    return a