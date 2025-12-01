import torch
import triton
import triton.language as tl

@triton.jit
def s174_kernel(
    a_ptr,
    b_ptr,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s174: a[i+M] = a[i] + b[i]
    Optimized for coalesced memory access and efficient vectorization.
    """
    # Get program ID and compute offset range
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for boundary checking
    mask = offsets < M
    
    # Load a[i] and b[i] with coalesced access
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute a[i] + b[i]
    result = a_vals + b_vals
    
    # Store to a[i+M] with coalesced access
    tl.store(a_ptr + offsets + M, result, mask=mask)

def s174_triton(a, b):
    """
    Triton implementation of TSVC s174.
    
    Optimizations:
    - Uses optimal block size (256) for memory coalescing
    - Single kernel launch with vectorized operations
    - Efficient masking for boundary conditions
    """
    a = a.contiguous()
    b = b.contiguous()
    
    M = b.size(0)
    
    # Optimal block size for memory coalescing and occupancy
    BLOCK_SIZE = 256
    grid = (triton.cdiv(M, BLOCK_SIZE),)
    
    # Launch kernel with optimized grid configuration
    s174_kernel[grid](
        a, b, M,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a