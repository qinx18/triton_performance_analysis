import torch
import triton
import triton.language as tl

@triton.jit
def s278_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and calculate offset range
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load all arrays with masking
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    d = tl.load(d_ptr + offsets, mask=mask)
    e = tl.load(e_ptr + offsets, mask=mask)
    
    # Create condition mask (a > 0)
    condition = a > 0.0
    
    # Compute d * e once (shared computation)
    de_product = d * e
    
    # Update b where condition is False (not taking goto L20)
    b_new = tl.where(condition, b, -b + de_product)
    
    # Update c where condition is True (taking goto L20)
    c_new = tl.where(condition, -c + de_product, c)
    
    # L30: Always execute - update a
    a_new = b_new + c_new * d
    
    # Store results back to memory
    tl.store(a_ptr + offsets, a_new, mask=mask)
    tl.store(b_ptr + offsets, b_new, mask=mask)
    tl.store(c_ptr + offsets, c_new, mask=mask)

def s278_triton(a, b, c, d, e):
    """
    Triton implementation of TSVC s278 - conditional assignments with goto logic.
    Optimized with vectorized conditional operations and coalesced memory access.
    """
    # Ensure tensors are contiguous and get common properties
    a = a.contiguous()
    b = b.contiguous() 
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()
    
    n_elements = a.numel()
    
    # Choose block size for optimal memory throughput
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel - operates in-place on input tensors
    s278_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a, b, c