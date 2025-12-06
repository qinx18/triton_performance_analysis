import torch
import triton
import triton.language as tl

@triton.jit
def s278_kernel(a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load arrays
    a = tl.load(a_ptr + idx, mask=mask)
    b = tl.load(b_ptr + idx, mask=mask)
    c = tl.load(c_ptr + idx, mask=mask)
    d = tl.load(d_ptr + idx, mask=mask)
    e = tl.load(e_ptr + idx, mask=mask)
    
    # Implement conditional logic: if a[i] > 0, update c; else update b
    condition = a > 0.0
    
    # When a[i] <= 0: b[i] = -b[i] + d[i] * e[i]
    new_b = tl.where(condition, b, -b + d * e)
    
    # When a[i] > 0: c[i] = -c[i] + d[i] * e[i]
    new_c = tl.where(condition, -c + d * e, c)
    
    # Final computation: a[i] = b[i] + c[i] * d[i]
    new_a = new_b + new_c * d
    
    # Store results
    tl.store(a_ptr + idx, new_a, mask=mask)
    tl.store(b_ptr + idx, new_b, mask=mask)
    tl.store(c_ptr + idx, new_c, mask=mask)

def s278_triton(a, b, c, d, e):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s278_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )