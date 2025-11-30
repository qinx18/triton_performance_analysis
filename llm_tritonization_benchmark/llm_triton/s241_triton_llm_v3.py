import triton
import triton.language as tl
import torch

@triton.jit
def s241_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values for first computation: a[i] = b[i] * c[i] * d[i]
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # Compute a[i] = b[i] * c[i] * d[i]
    a_vals = b_vals * c_vals * d_vals
    
    # Store a[i]
    tl.store(a_ptr + offsets, a_vals, mask=mask)
    
    # Load a[i+1] for second computation
    offsets_plus_1 = offsets + 1
    mask_plus_1 = offsets_plus_1 < (n_elements + 1)  # n_elements is already LEN_1D-1
    a_vals_plus_1 = tl.load(a_ptr + offsets_plus_1, mask=mask_plus_1, other=0.0)
    
    # Compute b[i] = a[i] * a[i+1] * d[i]
    b_new_vals = a_vals * a_vals_plus_1 * d_vals
    
    # Store b[i]
    tl.store(b_ptr + offsets, b_new_vals, mask=mask)

def s241_triton(a, b, c, d):
    n_elements = a.shape[0] - 1  # LEN_1D - 1
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s241_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )