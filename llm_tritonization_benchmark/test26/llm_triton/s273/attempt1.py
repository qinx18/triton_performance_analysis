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
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < n_elements
    
    # Load arrays
    a = tl.load(a_ptr + idx, mask=mask)
    d = tl.load(d_ptr + idx, mask=mask)
    e = tl.load(e_ptr + idx, mask=mask)
    b = tl.load(b_ptr + idx, mask=mask)
    
    # a[i] += d[i] * e[i]
    de_product = d * e
    a = a + de_product
    
    # if (a[i] < 0.) b[i] += d[i] * e[i]
    negative_mask = a < 0.0
    b = tl.where(negative_mask, b + de_product, b)
    
    # c[i] += a[i] * d[i]
    c = tl.load(c_ptr + idx, mask=mask)
    c = c + a * d
    
    # Store results
    tl.store(a_ptr + idx, a, mask=mask)
    tl.store(b_ptr + idx, b, mask=mask)
    tl.store(c_ptr + idx, c, mask=mask)

def s273_triton(a, b, c, d, e):
    n_elements = a.shape[0]
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s273_kernel[grid](
        a, b, c, d, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )