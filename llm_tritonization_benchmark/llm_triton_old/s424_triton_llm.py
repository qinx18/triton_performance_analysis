import torch
import triton
import triton.language as tl

@triton.jit
def s424_kernel(
    a_ptr,
    flat_2d_array_ptr,
    xx_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s424 - vectorized array assignment with offset indexing.
    Computes xx[i+1] = flat_2d_array[i] + a[i] for i in [0, n_elements-1)
    """
    # Get program ID and compute block start
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Generate offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle boundary conditions
    mask = offsets < n_elements
    
    # Load input data with masking
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    flat_2d_vals = tl.load(flat_2d_array_ptr + offsets, mask=mask)
    
    # Compute result: flat_2d_array[i] + a[i]
    result = flat_2d_vals + a_vals
    
    # Store to xx[i+1] (offset by 1)
    tl.store(xx_ptr + offsets + 1, result, mask=mask)

def s424_triton(a, flat_2d_array, xx):
    """
    Triton implementation of s424 - Loop with array assignment using offset indexing.
    
    Optimizations:
    - Vectorized memory access with configurable block size
    - Coalesced memory reads and writes
    - Efficient masking for boundary handling
    """
    # Ensure tensors are contiguous
    a = a.contiguous()
    flat_2d_array = flat_2d_array.contiguous()
    xx = xx.contiguous()
    
    # Number of elements to process (LEN_1D - 1)
    n_elements = a.shape[0] - 1
    
    if n_elements <= 0:
        return xx
    
    # Configure block size for optimal memory coalescing
    BLOCK_SIZE = 256
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s424_kernel[grid](
        a,
        flat_2d_array,
        xx,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return xx