import torch
import triton
import triton.language as tl

@triton.jit
def s4121_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and calculate offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary conditions
    mask = offsets < n_elements
    
    # Load data with masking
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    
    # Compute f(b[i], c[i]) = b[i] * c[i] and add to a[i]
    result = a_vals + b_vals * c_vals
    
    # Store result with masking
    tl.store(a_ptr + offsets, result, mask=mask)

def s4121_triton(a, b, c):
    """
    Triton implementation of TSVC s4121.
    Optimized with coalesced memory access and vectorized operations.
    """
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    
    n_elements = a.numel()
    
    # Choose block size for optimal memory coalescing
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s4121_kernel[grid](
        a,
        b,
        c,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a