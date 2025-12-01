import torch
import triton
import triton.language as tl

@triton.jit
def s162_kernel(
    a_ptr, b_ptr, c_ptr,
    n_elements, k,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and calculate element indices
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid elements (i < n_elements - 1)
    mask = offsets < (n_elements - 1)
    
    # Load b[i] and c[i] with masking
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    
    # Calculate source indices for a[i + k]
    source_offsets = offsets + k
    source_mask = mask & (source_offsets < n_elements)
    
    # Load a[i + k] with masking
    a_source_vals = tl.load(a_ptr + source_offsets, mask=source_mask, other=0.0)
    
    # Compute result: a[i + k] + b[i] * c[i]
    result = a_source_vals + b_vals * c_vals
    
    # Store result back to a[i] with masking
    tl.store(a_ptr + offsets, result, mask=mask)

def s162_triton(a, b, c, k):
    """
    Triton implementation of TSVC s162.
    Optimized with coalesced memory access and parallel computation.
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    
    if k <= 0 or k >= len(a):
        return a
    
    n_elements = len(a)
    
    # Use block size optimized for memory bandwidth
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements - 1, BLOCK_SIZE),)
    
    # Launch kernel with optimized block size for memory coalescing
    s162_kernel[grid](
        a, b, c,
        n_elements, k,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a