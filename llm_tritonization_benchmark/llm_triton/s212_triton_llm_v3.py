import triton
import triton.language as tl
import torch

@triton.jit
def s212_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, a_temp_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # First statement: a[i] *= c[i]
    a_vals = tl.load(a_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    a_new = a_vals * c_vals
    tl.store(a_ptr + offsets, a_new, mask=mask)
    tl.store(a_temp_ptr + offsets, a_new, mask=mask)
    
    # Second statement: b[i] += a[i + 1] * d[i]
    # Need to read a[i+1] values
    offsets_plus_one = offsets + 1
    mask_plus_one = offsets_plus_one < (n_elements + 1)
    a_next_vals = tl.load(a_temp_ptr + offsets_plus_one, mask=mask_plus_one)
    
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    b_new = b_vals + a_next_vals * d_vals
    tl.store(b_ptr + offsets, b_new, mask=mask)

def s212_triton(a, b, c, d):
    n_elements = a.shape[0] - 1
    
    # Create temporary array for WAR dependency handling
    a_temp = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s212_kernel[grid](
        a, b, c, d, a_temp,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return a, b