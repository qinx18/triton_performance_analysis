import torch
import triton
import triton.language as tl

@triton.jit
def s000_kernel(
    a_ptr,
    b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for vector addition with scalar (a[i] = b[i] + 1)
    Uses coalesced memory access and masking for edge cases
    """
    # Get program ID and compute offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for edge cases
    mask = offsets < n_elements
    
    # Load b values with masking
    b = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute a = b + 1
    a = b + 1
    
    # Store result with masking
    tl.store(a_ptr + offsets, a, mask=mask)

def s000_triton(a, b, iterations):
    """
    TSVC s000 - Simple vector addition with scalar using Triton
    Optimized with coalesced memory access and appropriate block sizing
    """
    a = a.contiguous()
    b = b.contiguous()
    
    n_elements = a.numel()
    
    # Choose block size for optimal memory coalescing
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s000_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a