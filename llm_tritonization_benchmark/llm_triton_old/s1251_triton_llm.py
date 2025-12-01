import torch
import triton
import triton.language as tl

@triton.jit
def s1251_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate block start position
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary checking
    mask = offsets < n_elements
    
    # Load input vectors with masking
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Compute intermediate value s = b + c
    s_vals = b_vals + c_vals
    
    # Compute new b = a + d
    new_b_vals = a_vals + d_vals
    
    # Compute new a = s * e
    new_a_vals = s_vals * e_vals
    
    # Store results with masking
    tl.store(a_ptr + offsets, new_a_vals, mask=mask)
    tl.store(b_ptr + offsets, new_b_vals, mask=mask)

def s1251_triton(a, b, c, d, e):
    """
    Triton implementation of TSVC s1251
    Optimized with coalesced memory access and vectorized operations
    """
    # Ensure tensors are contiguous for optimal memory access
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()
    
    n_elements = a.numel()
    
    # Choose block size for good occupancy and memory coalescing
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel with optimal block size
    s1251_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a, b