import triton
import triton.language as tl
import torch

@triton.jit
def s243_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, a_copy_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    
    mask = idx < n_elements
    
    # Load arrays
    b_vals = tl.load(b_ptr + idx, mask=mask)
    c_vals = tl.load(c_ptr + idx, mask=mask)
    d_vals = tl.load(d_ptr + idx, mask=mask)
    e_vals = tl.load(e_ptr + idx, mask=mask)
    
    # Load a[i+1] values from copy (for third statement)
    idx_plus_1 = idx + 1
    mask_plus_1 = idx_plus_1 < (n_elements + 1)  # Allow reading a[i+1] where i < n_elements-1
    a_plus_1_vals = tl.load(a_copy_ptr + idx_plus_1, mask=mask_plus_1, other=0.0)
    
    # First statement: a[i] = b[i] + c[i] * d[i]
    a_vals = b_vals + c_vals * d_vals
    tl.store(a_ptr + idx, a_vals, mask=mask)
    
    # Second statement: b[i] = a[i] + d[i] * e[i]
    b_new_vals = a_vals + d_vals * e_vals
    tl.store(b_ptr + idx, b_new_vals, mask=mask)
    
    # Third statement: a[i] = b[i] + a[i+1] * d[i]
    a_final_vals = b_new_vals + a_plus_1_vals * d_vals
    tl.store(a_ptr + idx, a_final_vals, mask=mask)

def s243_triton(a, b, c, d, e):
    n_elements = a.shape[0] - 1  # Loop runs for LEN_1D-1 iterations
    
    # Create read-only copy of array 'a' to handle WAR dependencies
    a_copy = a.clone()
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    s243_kernel[grid](
        a, b, c, d, e, a_copy,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )