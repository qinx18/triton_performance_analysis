import torch
import triton
import triton.language as tl

@triton.jit
def s211_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, b_orig_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate thread block starting position
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Generate offsets for the current block (indices 1 to n_elements-2)
    offsets = block_start + tl.arange(0, BLOCK_SIZE) + 1
    
    # Create mask to handle boundary conditions
    mask = (offsets >= 1) & (offsets < n_elements - 1)
    
    # Load data with masking for safe memory access
    b_prev = tl.load(b_orig_ptr + offsets - 1, mask=mask)
    b_next = tl.load(b_orig_ptr + offsets + 1, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Compute updates: vectorized arithmetic operations
    # a[i] = b[i-1] + c[i] * d[i]
    a_new = b_prev + c_vals * d_vals
    # b[i] = b[i+1] - e[i] * d[i]  
    b_new = b_next - e_vals * d_vals
    
    # Store results with masking
    tl.store(a_ptr + offsets, a_new, mask=mask)
    tl.store(b_ptr + offsets, b_new, mask=mask)

def s211_triton(a, b, c, d, e):
    """
    Triton-optimized implementation of TSVC s211 function.
    
    Key optimizations:
    - Vectorized memory access patterns
    - Coalesced reads/writes for better memory bandwidth
    - Block-based processing for optimal GPU utilization
    """
    # Ensure tensors are contiguous for optimal memory access
    a = a.contiguous()
    b = b.contiguous()
    c = c.contiguous()
    d = d.contiguous()
    e = e.contiguous()
    
    n_elements = a.numel()
    
    # Store original b values before modification
    b_orig = b.clone()
    
    # Configure block size for optimal memory coalescing
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements - 2, BLOCK_SIZE),)
    
    # Launch kernel with optimal grid configuration
    s211_kernel[grid](
        a, b, c, d, e, b_orig,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a, b