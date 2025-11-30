import torch
import triton
import triton.language as tl

@triton.jit
def s2244_kernel(a_ptr, b_ptr, c_ptr, e_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    e_vals = tl.load(e_ptr + offsets, mask=mask)
    
    # Compute values
    val1 = b_vals + e_vals  # for a[i+1]
    val2 = b_vals + c_vals  # for a[i]
    
    # Store a[i+1] = b[i] + e[i] (with offset +1)
    mask_plus1 = (offsets + 1) < (n_elements + 1)
    tl.store(a_ptr + offsets + 1, val1, mask=mask & mask_plus1)
    
    # Store a[i] = b[i] + c[i]
    tl.store(a_ptr + offsets, val2, mask=mask)

def s2244_triton(a, b, c, e):
    n_elements = len(b) - 1  # Loop goes to LEN_1D-1
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s2244_kernel[grid](
        a, b, c, e,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )