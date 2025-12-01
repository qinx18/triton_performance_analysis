import torch
import triton
import triton.language as tl

@triton.jit
def s443_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate block start position
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load data with masking
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    d = tl.load(d_ptr + offsets, mask=mask)
    
    # Conditional logic: if d[i] <= 0, use b*c, else use b*b
    condition = d <= 0.0
    update = tl.where(condition, b * c, b * b)
    
    # Update a[i] += update
    result = a + update
    
    # Store result with masking
    tl.store(a_ptr + offsets, result, mask=mask)

def s443_triton(a, b, c, d):
    """
    Triton implementation of TSVC s443 - conditional assignment optimization
    Uses vectorized conditional operations and coalesced memory access
    """
    # Ensure tensors are contiguous
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    
    n_elements = a.numel()
    
    # Choose block size for optimal memory coalescing
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel with flattened tensors
    s443_kernel[grid](
        a.view(-1), b.view(-1), c.view(-1), d.view(-1),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a