import torch
import triton
import triton.language as tl

@triton.jit
def s272_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    t,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for conditional updates with arithmetic operations.
    Uses vectorized loads and conditional masking for optimal performance.
    """
    # Get program ID and calculate offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Boundary mask to handle non-multiple-of-block-size arrays
    mask = offsets < n_elements
    
    # Load input arrays with masking
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    d = tl.load(d_ptr + offsets, mask=mask)
    e = tl.load(e_ptr + offsets, mask=mask)
    
    # Conditional mask: e[i] >= t
    cond_mask = e >= t
    
    # Compute updates
    c_times_d = c * d
    c_squared = c * c
    
    # Apply conditional updates using tl.where
    a_new = tl.where(cond_mask, a + c_times_d, a)
    b_new = tl.where(cond_mask, b + c_squared, b)
    
    # Store results with boundary masking
    tl.store(a_ptr + offsets, a_new, mask=mask)
    tl.store(b_ptr + offsets, b_new, mask=mask)

def s272_triton(a, b, c, d, e, t):
    """
    Triton implementation of TSVC s272 - conditional updates with arithmetic operations.
    Optimized with vectorized operations and efficient memory access patterns.
    """
    # Ensure tensors are contiguous and get output tensors
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()
    
    # Create output tensors (copy inputs since we need to preserve original values)
    a_out = a.clone()
    b_out = b.clone()
    
    n_elements = a.numel()
    
    # Choose block size based on problem size and memory constraints
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    s272_kernel[grid](
        a_out, b_out, c, d, e,
        t,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a_out, b_out