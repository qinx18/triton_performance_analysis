import torch
import triton
import triton.language as tl

@triton.jit
def s131_kernel(a_ptr, b_ptr, m, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for s131 operation.
    Each thread processes one element: a[i] = a[i + m] + b[i]
    """
    # Get program ID and calculate element index
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements (i < n_elements - 1)
    mask = offsets < n_elements - 1
    
    # Create mask for valid memory access (i + m < total array size)
    source_offsets = offsets + m
    source_mask = mask & (source_offsets < n_elements)
    
    # Load data with masking
    a_source = tl.load(a_ptr + source_offsets, mask=source_mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Compute result
    result = a_source + b_vals
    
    # Store result back to a[i]
    tl.store(a_ptr + offsets, result, mask=mask)

def s131_triton(a, b, m):
    """
    Triton implementation of TSVC s131 function.
    Optimized with parallel processing and coalesced memory access.
    """
    a = a.contiguous()
    b = b.contiguous()
    
    n_elements = a.size(0)
    
    # Choose block size for optimal memory coalescing
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements - 1, BLOCK_SIZE),)
    
    # Launch kernel
    s131_kernel[grid](
        a, b, m, n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a