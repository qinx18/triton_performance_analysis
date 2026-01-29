import torch
import triton
import triton.language as tl

@triton.jit
def s1251_expand_scalar_kernel(b_ptr, c_ptr, s_expanded_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
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
    offsets = tl.arange(0, BLOCK_SIZE)
    indices = block_start + offsets
    
    mask = indices < n_elements
    
    # Load expanded scalar values
    s_vals = tl.load(s_expanded_ptr + indices, mask=mask)
    
    # Load arrays
    a_vals = tl.load(a_ptr + indices, mask=mask)
    d_vals = tl.load(d_ptr + indices, mask=mask)
    e_vals = tl.load(e_ptr + indices, mask=mask)
    
    # Compute: b[i] = a[i] + d[i]
    b_new = a_vals + d_vals
    tl.store(b_ptr + indices, b_new, mask=mask)
    
    # Compute: a[i] = s * e[i]
    a_new = s_vals * e_vals
    tl.store(a_ptr + indices, a_new, mask=mask)

def s1251_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    
    # Create expanded scalar array
    s_expanded = torch.zeros_like(a)
    
    # Phase 1: Expand scalar s
    grid = (1,)  # Single block
    s1251_expand_scalar_kernel[grid](
        b, c, s_expanded, n_elements, BLOCK_SIZE
    )
    
    # Phase 2: Parallel computation using expanded scalar
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    s1251_kernel[grid](
        a, b, c, d, e, s_expanded, n_elements, BLOCK_SIZE
    )