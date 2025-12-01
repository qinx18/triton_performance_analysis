import torch
import triton
import triton.language as tl

@triton.jit
def s273_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load arrays
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = tl.load(c_ptr + offsets, mask=mask)
    d = tl.load(d_ptr + offsets, mask=mask)
    e = tl.load(e_ptr + offsets, mask=mask)
    
    # a[i] += d[i] * e[i]
    de = d * e
    a = a + de
    
    # if (a[i] < 0.) b[i] += d[i] * e[i]
    condition = a < 0.0
    b = tl.where(condition, b + de, b)
    
    # c[i] += a[i] * d[i]
    c = c + a * d
    
    # Store results
    tl.store(a_ptr + offsets, a, mask=mask)
    tl.store(b_ptr + offsets, b, mask=mask)
    tl.store(c_ptr + offsets, c, mask=mask)

def s273_triton(a, b, c, d, e):
    n_elements = a.numel()
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s273_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )