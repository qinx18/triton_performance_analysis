import triton
import triton.language as tl
import torch

@triton.jit
def s244_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID and compute offsets
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for main loop (i < n_elements - 1)
    mask = offsets < (n_elements - 1)
    
    # Load values
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b_vals = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    c_vals = tl.load(c_ptr + offsets, mask=mask, other=0.0)
    d_vals = tl.load(d_ptr + offsets, mask=mask, other=0.0)
    
    # Load a[i+1] values (need mask for i+1 < n_elements)
    next_offsets = offsets + 1
    next_mask = next_offsets < n_elements
    a_next_vals = tl.load(a_ptr + next_offsets, mask=next_mask, other=0.0)
    
    # Execute statements in order
    # S0: a[i] = b[i] + c[i] * d[i]
    a_new = b_vals + c_vals * d_vals
    
    # S1: b[i] = c[i] + b[i]
    b_new = c_vals + b_vals
    
    # S2: a[i+1] = b[i] + a[i+1] * d[i] (use updated b[i])
    a_next_new = b_new + a_next_vals * d_vals
    
    # Store results
    tl.store(a_ptr + offsets, a_new, mask=mask)
    tl.store(b_ptr + offsets, b_new, mask=mask)
    tl.store(a_ptr + next_offsets, a_next_new, mask=(mask & next_mask))

def s244_triton(a, b, c, d):
    n_elements = a.shape[0]
    
    BLOCK_SIZE = 256
    grid_size = triton.cdiv(n_elements, BLOCK_SIZE)
    
    s244_kernel[(grid_size,)](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a, b, c, d