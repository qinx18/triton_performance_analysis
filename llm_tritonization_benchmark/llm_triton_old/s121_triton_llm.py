import torch
import triton
import triton.language as tl

@triton.jit
def s121_kernel(
    a_ptr,
    b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s121: a[i] = a[i+1] + b[i] for i in [0, n_elements-1)
    
    Key optimizations:
    - Vectorized memory access with BLOCK_SIZE elements per thread block
    - Coalesced memory reads for a[i+1] and b[i]
    - Masked operations to handle non-aligned array sizes
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid indices (i < n_elements)
    mask = offsets < n_elements
    
    # Load a[i+1] and b[i] with bounds checking
    a_next_offsets = offsets + 1
    a_next_mask = a_next_offsets < (n_elements + 1)  # a has one more element than we're processing
    
    a_next = tl.load(a_ptr + a_next_offsets, mask=mask & a_next_mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    
    # Compute a[i] = a[i+1] + b[i]
    result = a_next + b_vals
    
    # Store result back to a[i]
    tl.store(a_ptr + offsets, result, mask=mask)

def s121_triton(a, b):
    """
    Triton implementation of TSVC s121 function.
    
    Optimizations:
    - Single kernel launch processes all elements
    - Block size tuned for memory bandwidth
    - Handles edge case of arrays with length <= 1
    """
    if len(a) <= 1:
        return a
    
    # Number of elements to process (LEN_1D - 1)
    n_elements = len(a) - 1
    
    # Choose block size for good occupancy
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s121_kernel[grid](
        a,
        b, 
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a