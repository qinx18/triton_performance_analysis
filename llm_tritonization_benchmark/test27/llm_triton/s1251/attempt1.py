import torch
import triton
import triton.language as tl

@triton.jit
def s1251_expand_s_kernel(b_ptr, c_ptr, s_expanded_ptr, n_elements):
    pid = tl.program_id(0)
    if pid != 0:
        return
    
    s_val = 0.0
    for i in range(n_elements):
        b_val = tl.load(b_ptr + i)
        c_val = tl.load(c_ptr + i)
        s_val = b_val + c_val
        tl.store(s_expanded_ptr + i, s_val)

@triton.jit
def s1251_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    mask = indices < n_elements
    
    s_vals = tl.load(s_expanded_ptr + indices, mask=mask)
    a_vals = tl.load(a_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    e_vals = tl.load(e_ptr + indices, mask=mask)
    
    b_new = a_vals + d_vals
    a_new = s_vals * e_vals
    
    tl.store(b_ptr + indices, b_new, mask=mask)
    tl.store(a_ptr + indices, a_new, mask=mask)

def s1251_triton(a, b, c, d, e):
    n = a.shape[0]
    BLOCK_SIZE = 256
    
    s_expanded = torch.zeros(n, dtype=a.dtype, device=a.device)
    
    grid = (1,)
    s1251_expand_s_kernel[grid](b, c, s_expanded, n)
    
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    s1251_kernel[grid](a, b, c, d, e, s_expanded, n, BLOCK_SIZE)