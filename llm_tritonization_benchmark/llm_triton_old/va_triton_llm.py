import torch
import triton
import triton.language as tl

@triton.jit
def va_kernel(
    a_ptr,
    b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the starting position for this block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for the current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle elements beyond array bounds
    mask = offsets < n_elements
    
    # Load data from b array with masking
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    
    # Store data to a array with masking
    tl.store(a_ptr + offsets, b_vals, mask=mask)

def va_triton(a, b):
    """
    Triton implementation of TSVC function va.
    Optimized vector copy operation using GPU parallelization.
    """
    # Ensure tensors are contiguous for optimal memory access
    a = a.contiguous()
    b = b.contiguous()
    
    n_elements = a.numel()
    
    # Use power-of-2 block size for optimal memory coalescing
    BLOCK_SIZE = 1024
    
    # Calculate grid size to cover all elements
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel with optimized block size
    va_kernel[grid](
        a, b, n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a