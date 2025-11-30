import triton
import triton.language as tl
import torch

@triton.jit
def s244_kernel(a_ptr, b_ptr, c_ptr, d_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load values for current iteration
    b_vals = tl.load(b_ptr + offsets, mask=mask)
    c_vals = tl.load(c_ptr + offsets, mask=mask)
    d_vals = tl.load(d_ptr + offsets, mask=mask)
    
    # Load a[i+1] values (shift by 1)
    offsets_plus_1 = offsets + 1
    mask_plus_1 = offsets_plus_1 < (n_elements + 1)
    a_plus_1_vals = tl.load(a_ptr + offsets_plus_1, mask=mask_plus_1)
    
    # First statement: a[i] = b[i] + c[i] * d[i]
    a_vals = b_vals + c_vals * d_vals
    
    # Second statement: b[i] = c[i] + b[i]
    b_new_vals = c_vals + b_vals
    
    # Third statement: a[i+1] = b[i] + a[i+1] * d[i]
    # Use the updated b values from statement 2
    a_plus_1_new_vals = b_new_vals + a_plus_1_vals * d_vals
    
    # Store results
    tl.store(a_ptr + offsets, a_vals, mask=mask)
    tl.store(b_ptr + offsets, b_new_vals, mask=mask)
    tl.store(a_ptr + offsets_plus_1, a_plus_1_new_vals, mask=mask_plus_1)

def s244_triton(a, b, c, d):
    n_elements = a.shape[0] - 1  # Loop goes to LEN_1D-1
    
    # Choose block size
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s244_kernel[grid](
        a, b, c, d,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )