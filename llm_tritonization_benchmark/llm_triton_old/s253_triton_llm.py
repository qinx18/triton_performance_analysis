import torch
import triton
import triton.language as tl

@triton.jit
def s253_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, a_out_ptr, c_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate starting position for this block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements to handle edge cases
    mask = offsets < n_elements
    
    # Load input data with masking for out-of-bounds protection
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    d = tl.load(d_ptr + offsets, mask=mask)
    
    # Compute condition mask: a > b
    condition = a > b
    
    # Compute s = a - b * d for all elements
    s = a - b * d
    
    # Apply conditional logic using tl.where
    a_out = tl.where(condition, s, a)
    c_out = tl.where(condition, c + s, c)
    
    # Store results with masking
    tl.store(a_out_ptr + offsets, a_out, mask=mask)
    tl.store(c_out_ptr + offsets, c_out, mask=mask)

def s253_triton(a, b, c, d):
    """
    Triton implementation of TSVC s253 function.
    Optimized for GPU execution with coalesced memory access.
    """
    # Ensure contiguous memory layout for optimal performance
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    
    n_elements = a.numel()
    
    # Allocate output tensors
    a_out = torch.empty_like(a)
    c_out = torch.empty_like(c)
    
    # Choose block size for optimal memory coalescing
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel with optimized block size
    s253_kernel[grid](
        a, b, c, d, a_out, c_out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a_out, c_out