import torch
import triton
import triton.language as tl

@triton.jit
def s453_kernel(
    a_ptr,
    b_ptr,
    len_1d,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s453: a[i] = (2 * (i + 1)) * b[i]
    Computes cumulative sum of 2's multiplied by b values
    """
    # Get program ID and compute block start
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offset range for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle boundary conditions
    mask = offsets < len_1d
    
    # Load b values with masking
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Compute s values: s = 2 * (i + 1) = 2, 4, 6, 8, ...
    # offsets are 0-based indices, so add 1 then multiply by 2
    s_vals = (offsets + 1) * 2.0
    
    # Compute result: a[i] = s * b[i]
    result = s_vals * b_vals
    
    # Store result with masking
    tl.store(a_ptr + offsets, result, mask=mask)

def s453_triton(a, b):
    """
    Triton implementation of TSVC s453.
    Optimized with coalesced memory access and efficient vectorization.
    """
    a = a.contiguous()
    b = b.contiguous()
    
    len_1d = a.size(0)
    
    # Choose block size for optimal memory coalescing
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(len_1d, BLOCK_SIZE),)
    
    # Launch kernel
    s453_kernel[grid](
        a, b, len_1d,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a