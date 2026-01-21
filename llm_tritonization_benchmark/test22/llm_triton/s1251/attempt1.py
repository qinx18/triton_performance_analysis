import torch
import triton
import triton.language as tl

@triton.jit
def s1251_expand_s_kernel(b_ptr, c_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    if pid == 0:  # Single thread processes all elements sequentially
        s_val = 0.0
        for i in range(n_elements):
            b_val = tl.load(b_ptr + i)
            c_val = tl.load(c_ptr + i)
            s_val = b_val + c_val
            tl.store(s_expanded_ptr + i, s_val)

@triton.jit
def s1251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    s_vals = tl.load(s_expanded_ptr + offsets, mask=mask)
    
    # Compute
    new_b_vals = a_vals + d_vals
    new_a_vals = s_vals * e_vals
    
    # Store results
    tl.store(b_ptr + offsets, new_b_vals, mask=mask)
    tl.store(a_ptr + offsets, new_a_vals, mask=mask)

def s1251_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Allocate temporary array for scalar expansion
    s_expanded = torch.zeros_like(a)
    
    # Phase 1: Expand scalar s
    grid_expand = (1,)  # Single thread
    s1251_expand_s_kernel[grid_expand](
        b, c, s_expanded, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Phase 2: Parallel computation using expanded array
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s1251_kernel[grid](
        a, b, c, d, e, s_expanded, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return a, b