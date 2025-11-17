import torch
import triton
import triton.language as tl

@triton.jit
def s452_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get program ID and compute block start
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle edge cases
    mask = offsets < n_elements
    
    # Load data with masking
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    
    # Compute indices (1-based) - cast to float for computation
    indices = (offsets + 1).to(tl.float32)
    
    # Perform computation: a[i] = b[i] + c[i] * (i+1)
    result = b + c * indices
    
    # Store result with masking
    tl.store(a_ptr + offsets, result, mask=mask)

def s452_triton(a, b, c):
    """
    Triton implementation of TSVC s452
    Optimized with coalesced memory access and efficient blocking
    """
    # Ensure tensors are contiguous for optimal memory access
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    
    n_elements = a.numel()
    
    # Use power-of-2 block size for optimal memory coalescing
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s452_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a