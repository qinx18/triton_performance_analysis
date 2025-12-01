import torch
import triton
import triton.language as tl

@triton.jit
def s271_kernel(a_ptr, b_ptr, c_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the current block's starting position
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for the current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle edge cases where block extends beyond array bounds
    mask = offsets < n_elements
    
    # Load data from global memory with masking
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    
    # Apply conditional logic: if b[i] > 0, then a[i] += b[i] * c[i]
    condition = b_vals > 0.0
    result = tl.where(condition, a_vals + b_vals * c_vals, a_vals)
    
    # Store the result back to global memory
    tl.store(a_ptr + offsets, result, mask=mask)

def s271_triton(a, b, c):
    """
    Triton implementation of TSVC s271 function.
    Optimized with coalesced memory access and efficient conditional execution.
    """
    # Ensure tensors are contiguous for optimal memory access patterns
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    
    n_elements = a.numel()
    
    # Choose block size for good occupancy and memory coalescing
    BLOCK_SIZE = 1024
    
    # Calculate grid size (number of blocks needed)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch the kernel
    s271_kernel[grid](
        a, b, c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a