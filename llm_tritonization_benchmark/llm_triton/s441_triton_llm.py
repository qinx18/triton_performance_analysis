import torch
import triton
import triton.language as tl

@triton.jit
def s441_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for conditional linear recurrence with three branches
    Optimizes memory coalescing and reduces conditional overhead
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load all arrays with masking for edge cases
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    d = tl.load(d_ptr + offsets, mask=mask)
    
    # Create condition masks for the three branches
    mask_negative = d < 0.0
    mask_zero = d == 0.0
    mask_positive = ~(mask_negative | mask_zero)
    
    # Compute all three possible updates
    update_negative = b * c
    update_zero = b * b
    update_positive = c * c
    
    # Apply conditional updates using select operations for better performance
    result = tl.where(mask_negative, a + update_negative, a)
    result = tl.where(mask_zero, result + update_zero, result)
    result = tl.where(mask_positive, result + update_positive, result)
    
    # Store result back to memory
    tl.store(a_ptr + offsets, result, mask=mask)

def s441_triton(a, b, c, d):
    """
    Triton implementation of TSVC s441 - conditional linear recurrence
    Uses vectorized conditional operations for GPU acceleration
    """
    # Ensure tensors are contiguous and on the same device
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    
    n_elements = a.numel()
    
    # Choose block size for optimal memory coalescing
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel with flattened tensor views
    s441_kernel[grid](
        a.view(-1), b.view(-1), c.view(-1), d.view(-1),
        n_elements,
        BLOCK_SIZE,
    )
    
    return a