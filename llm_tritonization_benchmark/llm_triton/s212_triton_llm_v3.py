import triton
import triton.language as tl
import torch

@triton.jit
def s212_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load current elements
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # First statement: a[i] *= c[i]
    a_new = a_vals * c_vals
    tl.store(a_ptr + offsets, a_new, mask=mask)
    
    # Load a[i+1] for second statement
    offsets_plus_1 = offsets + 1
    mask_plus_1 = offsets_plus_1 < (n_elements + 1)
    a_plus_1 = tl.load(a_ptr + offsets_plus_1, mask=mask_plus_1)
    
    # Second statement: b[i] += a[i + 1] * d[i]
    b_new = b_vals + a_plus_1 * d_vals
    tl.store(b_ptr + offsets, b_new, mask=mask)

def s212_triton(a, b, c, d):
    n_elements = a.shape[0] - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s212_kernel[grid](a, b, c, d, n_elements, BLOCK_SIZE=BLOCK_SIZE)