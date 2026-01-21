import torch
import triton
import triton.language as tl

@triton.jit
def s1251_expand_s_kernel(b_ptr, c_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    # Single thread processes all elements sequentially for scalar expansion
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
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    s_vals = tl.load(s_expanded_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Compute: b[i] = a[i] + d[i]
    b_new = a_vals + d_vals
    
    # Compute: a[i] = s * e[i]
    a_new = s_vals * e_vals
    
    # Store results
    tl.store(b_ptr + offsets, b_new, mask=mask)
    tl.store(a_ptr + offsets, a_new, mask=mask)

def s1251_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create expanded scalar array
    s_expanded = torch.zeros(n_elements, dtype=a.dtype, device=a.device)
    
    # Phase 1: Expand scalar s
    grid = (1,)
    s1251_expand_s_kernel[grid](b, c, s_expanded, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    # Phase 2: Parallel computation using expanded scalar
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s1251_kernel[grid](a, b, c, d, e, s_expanded, n_elements, BLOCK_SIZE=BLOCK_SIZE)