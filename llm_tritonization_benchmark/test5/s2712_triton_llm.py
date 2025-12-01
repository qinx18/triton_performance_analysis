import torch
import triton
import triton.language as tl

@triton.jit
def s2712_kernel(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s2712: conditional multiply-add operation.
    Optimized with coalesced memory access and efficient masking.
    """
    # Get program ID and calculate offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements to handle edge cases
    mask = offsets < n_elements
    
    # Load data with masking for bounds checking
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    
    # Conditional computation: if a[i] > b[i], then a[i] += b[i] * c[i]
    condition = a > b
    result = tl.where(condition, a + b * c, a)
    
    # Store result back to memory
    tl.store(a_ptr + offsets, result, mask=mask)

def s2712_triton(a, b, c):
    """
    Triton implementation of TSVC s2712 function.
    Uses optimized GPU kernel with efficient memory coalescing.
    """
    # Ensure tensors are contiguous for optimal memory access
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    
    n_elements = a.numel()
    
    # Choose block size for optimal occupancy
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel with flattened tensors for efficient processing
    s2712_kernel[grid](
        a.view(-1), b.view(-1), c.view(-1),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a