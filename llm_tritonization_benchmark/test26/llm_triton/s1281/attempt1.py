import torch
import triton
import triton.language as tl

@triton.jit
def expand_x_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, x_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    if pid != 0:
        return
    
    x_val = 0.0
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
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    current_offsets = block_start + offsets
    
    mask = current_offsets < n_elements
    
    x_vals = tl.load(x_expanded_ptr + current_offsets, mask=mask)
    
    a_new = x_vals - 1.0
    b_new = x_vals
    
    tl.store(a_ptr + current_offsets, a_new, mask=mask)
    tl.store(b_ptr + current_offsets, b_new, mask=mask)

def s1281_triton(a, b, c, d, e, x):
    n_elements = a.shape[0]
    
    x_expanded = torch.zeros_like(a)
    
    BLOCK_SIZE = 256
    
    # Phase 1: Expand scalar x
    grid = (1,)
    expand_x_kernel[grid](a, b, c, d, e, x_expanded, n_elements, BLOCK_SIZE)
    
    # Phase 2: Parallel computation
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s1281_kernel[grid](a, b, c, d, e, x_expanded, n_elements, BLOCK_SIZE)