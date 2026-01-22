import torch
import triton
import triton.language as tl

@triton.jit
def expand_x_kernel(b_ptr, c_ptr, a_ptr, d_ptr, e_ptr, x_expanded_ptr, n_elements):
    # Single thread processes all elements sequentially
    x_val = 0.0  # Initial value
    for i in range(n_elements):
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        a_val = tl.load(a_ptr + i)
        d_val = tl.load(d_ptr + i)
        e_val = tl.load(e_ptr + i)
        
        x_val = b_val * c_val + a_val * d_val + e_val
        tl.store(x_expanded_ptr + i, x_val)

@triton.jit
def s1281_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, x_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load expanded x values
    x_vals = tl.load(x_expanded_ptr + offsets, mask=mask)
    
    # Update arrays
    a_vals = x_vals - 1.0
    b_vals = x_vals
    
    # Store results
    tl.store(a_ptr + offsets, a_vals, mask=mask)
    tl.store(b_ptr + offsets, b_vals, mask=mask)

def s1281_triton(a, b, c, d, e, x):
    n_elements = a.shape[0]
    
    # Create expanded array for x
    x_expanded = torch.zeros_like(a)
    
    # Phase 1: Expand scalar x using single thread
    expand_x_kernel[(1,)](
        b, c, a, d, e, x_expanded, n_elements
    )
    
    # Phase 2: Use expanded array in parallel computation
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s1281_kernel[grid](
        a, b, c, d, e, x_expanded, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )