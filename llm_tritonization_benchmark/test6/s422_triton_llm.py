import torch
import triton
import triton.language as tl

@triton.jit
def s422_kernel(a_ptr, flat_2d_array_ptr, xx_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel for s422: xx[i] = flat_2d_array[i + 8] + a[i]
    Uses coalesced memory access and efficient vectorization.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for boundary conditions
    mask = offsets < n_elements
    
    # Load data with coalesced access patterns
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    flat_2d_vals = tl.load(flat_2d_array_ptr + offsets + 8, mask=mask)
    
    # Compute result
    result = flat_2d_vals + a_vals
    
    # Store result
    tl.store(xx_ptr + offsets, result, mask=mask)

def s422_triton(a, flat_2d_array, xx):
    """
    Triton implementation of TSVC s422.
    Optimized with block-level parallelization and coalesced memory access.
    """
    # Ensure tensors are contiguous for optimal memory access
    a = a.contiguous()
    flat_2d_array = flat_2d_array.contiguous()
    xx = xx.contiguous()
    
    n_elements = a.numel()
    
    # Use block size optimized for GPU memory hierarchy
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel with optimal grid configuration
    s422_kernel[grid](
        a, flat_2d_array, xx,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return xx