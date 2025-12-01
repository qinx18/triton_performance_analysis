import torch
import triton
import triton.language as tl

@triton.jit
def s471_kernel(
    b_ptr, c_ptr, d_ptr, e_ptr, x_ptr,
    b_out_ptr, x_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and calculate element indices
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary checking
    mask = offsets < n_elements
    
    # Load input data with masking
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Compute operations: vectorized d*d and d*e multiplications
    d_squared = d_vals * d_vals
    d_times_e = d_vals * e_vals
    
    # Calculate output values
    x_out = b_vals + d_squared
    b_out = c_vals + d_times_e
    
    # Store results with masking
    tl.store(x_out_ptr + offsets, x_out, mask=mask)
    tl.store(b_out_ptr + offsets, b_out, mask=mask)

def s471_triton(b, c, d, e, x, m):
    """
    Triton implementation of TSVC s471 function.
    Optimized with vectorized operations and coalesced memory access.
    """
    # Ensure contiguous memory layout for optimal access patterns
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()
    x = x.contiguous()
    
    # Create output tensors
    b_new = b.clone()
    x_new = x.clone()
    
    # Use power-of-2 block size for better memory coalescing
    BLOCK_SIZE = 256
    grid = (triton.cdiv(m, BLOCK_SIZE),)
    
    # Launch kernel with optimized block size
    s471_kernel[grid](
        b, c, d, e, x,
        b_new, x_new,
        m,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return b_new, x_new