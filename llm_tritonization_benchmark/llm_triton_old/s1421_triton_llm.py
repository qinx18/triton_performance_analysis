import torch
import triton
import triton.language as tl

@triton.jit
def s1421_kernel(
    a_ptr, b_ptr, xx_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s1421: b[i] = xx[i] + a[i] for first half of array
    Uses coalesced memory access and vectorized operations
    """
    pid = tl.program_id(axis=0)
    
    # Calculate offset for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask to handle edge cases where block exceeds array bounds
    mask = offsets < n_elements
    
    # Load data with masking for safe memory access
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    xx_vals = tl.load(xx_ptr + offsets, mask=mask)
    
    # Vectorized addition operation
    result = xx_vals + a_vals
    
    # Store result with masking
    tl.store(b_ptr + offsets, result, mask=mask)

def s1421_triton(a, b, xx):
    """
    Triton implementation of TSVC s1421.
    Optimized for GPU with block-based processing and coalesced memory access.
    """
    a = a.contiguous()
    b = b.contiguous()
    xx = xx.contiguous()
    
    len_1d = a.size(0)
    half_len = len_1d // 2
    
    # Only process first half of the array
    if half_len == 0:
        return b
    
    # Choose block size for optimal memory coalescing
    BLOCK_SIZE = 256
    
    # Calculate grid size to cover all elements in first half
    grid = (triton.cdiv(half_len, BLOCK_SIZE),)
    
    # Launch kernel with pointers to first half of arrays
    s1421_kernel[grid](
        a, b, xx,
        half_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return b