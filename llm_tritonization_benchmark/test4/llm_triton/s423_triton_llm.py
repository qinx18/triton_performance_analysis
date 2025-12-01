import torch
import triton
import triton.language as tl

@triton.jit
def s423_kernel(
    flat_2d_array_ptr,
    xx_ptr,
    a_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for s423: flat_2d_array[i+1] = xx[i] + a[i]
    
    Key optimizations:
    - Coalesced memory access patterns
    - Vectorized operations using BLOCK_SIZE
    - Boundary masking for safe memory access
    """
    # Get program ID and calculate element range for this block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask for valid elements to prevent out-of-bounds access
    mask = offsets < n_elements
    
    # Load input data with masking
    xx_vals = tl.load(xx_ptr + offsets, mask=mask, other=0.0)
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Compute result: xx[i] + a[i]
    result = xx_vals + a_vals
    
    # Store to flat_2d_array[i+1] (offset by 1)
    output_offsets = offsets + 1
    output_mask = offsets < n_elements  # Same mask since we're within bounds
    tl.store(flat_2d_array_ptr + output_offsets, result, mask=output_mask)

def s423_triton(a, flat_2d_array, xx):
    """
    Triton implementation of TSVC s423 function.
    
    Equivalent to: flat_2d_array[1:] = xx[:-1] + a[:-1]
    """
    # Ensure tensors are contiguous for optimal memory access
    a = a.contiguous()
    flat_2d_array = flat_2d_array.contiguous()
    xx = xx.contiguous()
    
    # Number of elements to process (excluding last element)
    n_elements = a.shape[0] - 1
    
    if n_elements <= 0:
        return flat_2d_array
    
    # Choose block size for good occupancy
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Launch kernel with optimal grid configuration
    s423_kernel[(grid_size,)](
        flat_2d_array,
        xx,
        a,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return flat_2d_array