import triton
import triton.language as tl
import torch

@triton.jit
def s212_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    # Load current values
    a_vals = tl.load(a_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    b_vals = tl.load(b_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    
    # Load a[i+1] values
    idx_plus1 = idx + 1
    mask_plus1 = idx_plus1 < (n_elements + 1)
    a_plus1_vals = tl.load(a_ptr + idx_plus1, mask=mask_plus1)
    
    # First statement: a[i] *= c[i]
    a_new = a_vals * c_vals
    
    # Store updated a values
    tl.store(a_ptr + idx, a_new, mask=mask)
    
    # Second statement: b[i] += a[i + 1] * d[i] (using original a[i+1])
    b_new = b_vals + a_plus1_vals * d_vals
    
    # Store updated b values
    tl.store(b_ptr + idx, b_new, mask=mask)

def s212_triton(a, b, c, d):
    n_elements = a.shape[0] - 1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s212_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )